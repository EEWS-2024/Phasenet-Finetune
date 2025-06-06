#!/usr/bin/env python3
"""
Training script untuk PhaseNet Indonesia dengan Decoder-Only Fine-tuning
Hanya decoder yang dilatih, encoder dibekukan (frozen)
Kompatibel dengan model pretrained 190703-214543 (NCEDC dataset)
"""

import argparse
import os
import sys

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU

import tensorflow as tf
# Force CPU usage
tf.config.set_visible_devices([], 'GPU')  # Hide GPU from TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia_sliding import DataConfig_Indonesia_3000, DataReader_Indonesia_Sliding_Train
import json
import datetime
from tqdm import tqdm

def save_config(config, model_dir):
    """Save configuration to JSON file"""
    config_dict = {}
    for key in dir(config):
        if not key.startswith('_'):
            value = getattr(config, key)
            if not callable(value):
                config_dict[key] = value
    
    config_file = os.path.join(model_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config_file}")

def save_frozen_layers_info(frozen_vars, trainable_vars, model_dir):
    """Save information about frozen and trainable layers"""
    info_file = os.path.join(model_dir, 'frozen_layers.txt')
    
    with open(info_file, 'w') as f:
        f.write("=== DECODER-ONLY FINE-TUNING LAYER INFO ===\n\n")
        
        f.write(f"FROZEN LAYERS (Encoder): {len(frozen_vars)} variables\n")
        f.write("-" * 50 + "\n")
        for var in frozen_vars:
            f.write(f"❄️  {var.name} - Shape: {var.shape}\n")
        
        f.write(f"\nTRAINABLE LAYERS (Decoder): {len(trainable_vars)} variables\n")
        f.write("-" * 50 + "\n")
        for var in trainable_vars:
            f.write(f"🔥 {var.name} - Shape: {var.shape}\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write(f"  Total frozen parameters: {sum([np.prod(var.shape) for var in frozen_vars]):,}\n")
        f.write(f"  Total trainable parameters: {sum([np.prod(var.shape) for var in trainable_vars]):,}\n")
        f.write(f"  Percentage trainable: {len(trainable_vars)/(len(frozen_vars)+len(trainable_vars))*100:.1f}%\n")
    
    print(f"📋 Frozen layers info saved to: {info_file}")

def save_loss_history(train_losses, val_losses, model_dir):
    """Save training and validation loss history to CSV file"""
    max_epochs = max(len(train_losses), len(val_losses))
    
    train_padded = train_losses + [None] * (max_epochs - len(train_losses))
    val_padded = val_losses + [None] * (max_epochs - len(val_losses))
    
    loss_df = pd.DataFrame({
        'epoch': range(1, max_epochs + 1),
        'train_loss': train_padded,
        'val_loss': val_padded
    })
    
    csv_path = os.path.join(model_dir, 'training_history.csv')
    loss_df.to_csv(csv_path, index=False)
    print(f"📊 Training history saved to: {csv_path}")
    
    return loss_df

def plot_loss_curves(train_losses, val_losses, model_dir):
    """Create and save training/validation loss curves plot for decoder-only training"""
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss (Decoder-Only)', linewidth=2, markersize=4)
    plt.title('Decoder-Only Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot both losses together
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss (Decoder-Only)', linewidth=2, markersize=4)
    
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss (Decoder-Only)', linewidth=2, markersize=4)
    
    plt.title('Decoder-Only Fine-tuning: Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss curves plot saved to: {plot_path}")

def freeze_encoder_layers(sess):
    """Freeze encoder layers and return frozen and trainable variables"""
    all_vars = tf.compat.v1.global_variables()
    
    # Define patterns untuk encoder layers yang akan dibekukan
    encoder_patterns = [
        'Input/',           # Input layer 
        'DownConv_'         # All downsampling layers
    ]
    
    # Define patterns untuk decoder layers yang akan dilatih
    decoder_patterns = [
        'UpConv_',          # All upsampling layers
        'Output/'           # Output layer
    ]
    
    frozen_vars = []
    trainable_vars = []
    
    for var in all_vars:
        var_name = var.name
        
        # Skip optimizer variables
        skip_patterns = ['Adam', 'adam', 'Momentum', 'momentum', 'global_step', 
                        'beta1_power', 'beta2_power', 'learning_rate']
        
        if any(pattern in var_name for pattern in skip_patterns):
            continue
        
        # Check if it's an encoder layer (to be frozen)
        is_encoder = any(pattern in var_name for pattern in encoder_patterns)
        
        # Check if it's a decoder layer (to be trained)
        is_decoder = any(pattern in var_name for pattern in decoder_patterns)
        
        if is_encoder:
            frozen_vars.append(var)
        elif is_decoder:
            trainable_vars.append(var)
        else:
            # For batch norm variables, decide based on scope
            if 'moving_mean' in var_name or 'moving_variance' in var_name:
                if any(pattern in var_name for pattern in encoder_patterns):
                    frozen_vars.append(var)
                elif any(pattern in var_name for pattern in decoder_patterns):
                    trainable_vars.append(var)
            else:
                # Default: if unclear, make it trainable (but log it)
                print(f"⚠️  Unclear variable classification: {var_name} - making trainable")
                trainable_vars.append(var)
    
    print(f"\n🧊 ENCODER FREEZING SUMMARY:")
    print(f"  Frozen variables (Encoder): {len(frozen_vars)}")
    print(f"  Trainable variables (Decoder): {len(trainable_vars)}")
    print(f"  Percentage trainable: {len(trainable_vars)/(len(frozen_vars)+len(trainable_vars))*100:.1f}%")
    
    return frozen_vars, trainable_vars

def load_pretrained_model_selective(sess, pretrained_model_path, frozen_vars, trainable_vars):
    """Load pretrained model with selective loading for decoder-only training"""
    print(f"\n🔄 Loading pretrained model (decoder-only) from: {pretrained_model_path}")
    
    try:
        # Check if checkpoint exists
        ckpt = tf.train.latest_checkpoint(pretrained_model_path)
        if not ckpt:
            print(f"❌ No checkpoint found in {pretrained_model_path}")
            return False
        
        print(f"Found checkpoint: {ckpt}")
        
        # Get checkpoint variables
        checkpoint_vars = tf.train.list_variables(ckpt)
        checkpoint_var_names = {name for name, shape in checkpoint_vars}
        
        # Load ALL variables first (encoder + decoder)
        all_vars = frozen_vars + trainable_vars
        loaded_count = 0
        
        for var in all_vars:
            var_name = var.name.split(':')[0]
            
            if var_name in checkpoint_var_names:
                try:
                    checkpoint_value = tf.train.load_variable(ckpt, var_name)
                    
                    if var.shape.as_list() == list(checkpoint_value.shape):
                        # Load with appropriate scaling
                        if var in trainable_vars:
                            # For decoder variables that will be trained
                            if 'kernel' in var_name or 'weight' in var_name:
                                scaled_value = checkpoint_value * 0.5  # Light scaling untuk decoder
                            else:
                                scaled_value = checkpoint_value
                        else:
                            # For encoder variables that will be frozen
                            scaled_value = checkpoint_value  # No scaling for frozen layers
                        
                        sess.run(tf.compat.v1.assign(var, scaled_value))
                        loaded_count += 1
                        
                except Exception as e:
                    print(f"  ⚠️  Failed to load {var_name}: {e}")
        
        print(f"✅ Successfully loaded {loaded_count}/{len(all_vars)} variables")
        print(f"🧊 Encoder variables: loaded and will be FROZEN")
        print(f"🔥 Decoder variables: loaded and will be TRAINABLE")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        return False

def create_decoder_only_optimizer(learning_rate, trainable_vars, loss):
    """Create optimizer that only updates decoder variables"""
    
    # Create learning rate with decay
    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate_node = tf.compat.v1.train.exponential_decay(
        learning_rate=learning_rate,
        global_step=global_step,
        decay_steps=8,  # decay every 8 epochs
        decay_rate=0.95,
        staircase=True
    )
    
    # Create optimizer dengan scope khusus untuk decoder-only
    with tf.compat.v1.variable_scope("decoder_optimizer"):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_node)
        
        # IMPORTANT: Only compute gradients for trainable (decoder) variables
        gradients_and_vars = optimizer.compute_gradients(loss, var_list=trainable_vars)
        
        # Filter out None gradients
        filtered_gradients_and_vars = [(grad, var) for grad, var in gradients_and_vars if grad is not None]
        
        print(f"🎯 Optimizer will update {len(filtered_gradients_and_vars)} decoder variables")
        
        # Apply gradients only to decoder variables
        train_op = optimizer.apply_gradients(filtered_gradients_and_vars, global_step=global_step)
    
    return train_op, learning_rate_node, global_step

def train_fn(args, data_reader_train, data_reader_valid=None):
    """Training function untuk decoder-only fine-tuning"""
    
    # Create model directory with timestamp
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    model_dir = os.path.join(args.model_dir, f"decoder3000_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"🎯 Decoder-only model will be saved to: {model_dir}")
    
    # Create model config (same as pretrained model)
    config = ModelConfig(
        X_shape=data_reader_train.config.X_shape,  # [3000, 1, 3]
        Y_shape=data_reader_train.config.Y_shape,  # [3000, 1, 3]
        n_channel=data_reader_train.config.n_channel,
        n_class=data_reader_train.config.n_class,
        sampling_rate=data_reader_train.config.sampling_rate,
        dt=data_reader_train.config.dt,
        use_batch_norm=True,
        use_dropout=True,
        drop_rate=args.drop_rate,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        decay_step=args.decay_step,
        decay_rate=args.decay_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss_type=args.loss_type,
        weight_decay=args.weight_decay,
        summary=args.summary,
        save_interval=args.save_interval,
        class_weights=[1.0, 1.0, 1.0]
    )
    
    # Save configuration
    save_config(config, model_dir)
    
    # Create datasets
    train_dataset = data_reader_train.dataset(args.batch_size, shuffle=True, drop_remainder=True)
    
    # Create placeholders
    X_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + data_reader_train.config.X_shape, name='X_input')
    Y_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + data_reader_train.config.Y_shape, name='Y_target')
    fname_placeholder = tf.compat.v1.placeholder(tf.string, [None], name='fname_input')
    
    # Create model
    model = UNet(config=config, input_batch=(X_placeholder, Y_placeholder, fname_placeholder), mode='train')
    
    # Configure for CPU-only training
    cpu_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    cpu_config.allow_soft_placement = True
    cpu_config.log_device_placement = False
    
    with tf.compat.v1.Session(config=cpu_config) as sess:
        # Initialize variables PERTAMA
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Freeze encoder layers
        frozen_vars, trainable_vars = freeze_encoder_layers(sess)
        
        # Save frozen layers info
        save_frozen_layers_info(frozen_vars, trainable_vars, model_dir)
        
        # Load pretrained model SEBELUM membuat optimizer
        if not load_pretrained_model_selective(sess, args.pretrained_model_path, frozen_vars, trainable_vars):
            print("❌ Failed to load pretrained model. Exiting.")
            return
        
        # Create decoder-only optimizer SETELAH load model
        decoder_train_op, learning_rate_node, global_step = create_decoder_only_optimizer(
            args.learning_rate, trainable_vars, model.loss
        )
        
        # Initialize optimizer variables yang baru dibuat
        optimizer_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="decoder_optimizer")
        if optimizer_vars:
            sess.run(tf.compat.v1.variables_initializer(optimizer_vars))
            print(f"🔧 Initialized {len(optimizer_vars)} optimizer variables")
        
        # Create saver untuk decoder-only model (exclude optimizer vars dari original model)
        model_vars = [var for var in tf.compat.v1.global_variables() 
                      if not any(pattern in var.name for pattern in ['Adam', 'adam', 'beta1_power', 'beta2_power'])
                      or 'decoder_optimizer' in var.name]
        saver = tf.compat.v1.train.Saver(var_list=model_vars, max_to_keep=5)
        
        # Initialize training variables
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print(f"\n🚀 Starting decoder-only fine-tuning...")
        print(f"   📊 Training windows: {len(data_reader_train.sliding_windows):,}")
        print(f"   🎯 Batch size: {args.batch_size}")
        print(f"   🧊 Encoder: FROZEN ({len(frozen_vars)} variables)")
        print(f"   🔥 Decoder: TRAINABLE ({len(trainable_vars)} variables)")
        print(f"   ⚡ Expected speedup: ~{len(frozen_vars)/(len(trainable_vars)+1):.1f}x")
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
            
            # Get current learning rate
            current_lr = sess.run(learning_rate_node)
            print(f"   📉 Learning rate: {current_lr:.8f}")
            
            # Reset epoch metrics
            epoch_losses = []
            
            # Create dataset iterator
            train_iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
            next_train_batch = train_iterator.get_next()
            sess.run(train_iterator.initializer)
            
            # Calculate steps per epoch
            training_steps = int(np.ceil(len(data_reader_train.sliding_windows) / args.batch_size))
            
            # Training loop untuk epoch ini
            try:
                for step in range(training_steps):
                    try:
                        # Get batch
                        X_batch, Y_batch, fname_batch = sess.run(next_train_batch)
                        
                        # Train step (only decoder variables will be updated)
                        feed_dict = {
                            X_placeholder: X_batch,
                            Y_placeholder: Y_batch,
                            fname_placeholder: fname_batch,
                            model.drop_rate: args.drop_rate,
                            model.is_training: True
                        }
                        
                        # Run training step
                        _, loss_value = sess.run([decoder_train_op, model.loss], feed_dict=feed_dict)
                        epoch_losses.append(loss_value)
                        
                        # Print progress
                        if step % 50 == 0 or step == training_steps - 1:
                            avg_loss = np.mean(epoch_losses)
                            print(f"   Step {step+1}/{training_steps}, Loss: {loss_value:.6f}, Avg: {avg_loss:.6f}")
                        
                    except tf.errors.OutOfRangeError:
                        break
                        
            except Exception as e:
                print(f"⚠️  Training error at epoch {epoch+1}: {e}")
                break
            
            # Calculate epoch average
            if epoch_losses:
                epoch_avg_loss = np.mean(epoch_losses)
                train_losses.append(epoch_avg_loss)
                print(f"   ✅ Epoch {epoch+1} - Avg Training Loss: {epoch_avg_loss:.6f}")
                
                # Validation if available
                if data_reader_valid:
                    # TODO: Add validation logic here
                    pass
                
                # Save model periodically
                if (epoch + 1) % args.save_interval == 0:
                    save_path = os.path.join(model_dir, f"decoder_model_epoch_{epoch+1}.ckpt")
                    saver.save(sess, save_path)
                    print(f"   💾 Model saved: {save_path}")
            
            else:
                print(f"   ⚠️  No loss data for epoch {epoch+1}")
        
        # Final save
        final_save_path = os.path.join(model_dir, "decoder_model_final.ckpt")
        saver.save(sess, final_save_path)
        print(f"\n💾 Final decoder-only model saved: {final_save_path}")
        
        # Save training history and plots
        if train_losses:
            save_loss_history(train_losses, val_losses, model_dir)
            plot_loss_curves(train_losses, val_losses, model_dir)
            
            print(f"\n📊 DECODER-ONLY TRAINING COMPLETED")
            print(f"   Initial loss: {train_losses[0]:.6f}")
            print(f"   Final loss: {train_losses[-1]:.6f}")
            print(f"   Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
            print(f"   🎯 Only decoder was trained, encoder remained frozen")

def main():
    parser = argparse.ArgumentParser(description='Decoder-Only Fine-tuning PhaseNet Indonesia')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--train_list', type=str, required=True, help='Training data list CSV')
    parser.add_argument('--valid_dir', type=str, help='Validation data directory')
    parser.add_argument('--valid_list', type=str, help='Validation data list CSV')
    parser.add_argument('--format', type=str, default='numpy', help='Data format')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, default='model_indonesia/decoder_only', help='Model directory')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--log_dir', type=str, default='logs_indonesia/decoder_only', help='Log directory')
    
    # Training parameters optimized for decoder-only training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (less needed for decoder-only)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (can be larger for decoder-only)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate (higher for decoder)')
    parser.add_argument('--drop_rate', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--decay_step', type=int, default=8, help='Decay step')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    
    # Decoder-only specific parameters
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder layers (decoder-only training)')
    parser.add_argument('--summary', action='store_true', help='Enable summary')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.train_dir):
        print(f"❌ Training directory not found: {args.train_dir}")
        return
    
    if not os.path.exists(args.train_list):
        print(f"❌ Training list not found: {args.train_list}")
        return
    
    if not os.path.exists(args.pretrained_model_path):
        print(f"❌ Pretrained model not found: {args.pretrained_model_path}")
        return
    
    print("🎯 PhaseNet Indonesia Decoder-Only Fine-tuning")
    print("=" * 50)
    print(f"Train dir: {args.train_dir}")
    print(f"Train list: {args.train_list}")
    print(f"Pretrained model: {args.pretrained_model_path}")
    print(f"Output dir: {args.model_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Strategy: 🧊 Encoder FROZEN, 🔥 Decoder TRAINABLE")
    print("=" * 50)
    
    # Create data readers
    data_config = DataConfig_Indonesia_3000()
    
    print("Loading training data reader...")
    data_reader_train = DataReader_Indonesia_Sliding_Train(
        data_dir=args.train_dir,
        data_list=args.train_list,
        config=data_config,
        format=args.format
    )
    
    data_reader_valid = None
    if args.valid_dir and args.valid_list:
        print("Loading validation data reader...")
        data_reader_valid = DataReader_Indonesia_Sliding_Train(
            data_dir=args.valid_dir,
            data_list=args.valid_list,
            config=data_config,
            format=args.format
        )
    
    # Start decoder-only training
    train_fn(args, data_reader_train, data_reader_valid)

if __name__ == '__main__':
    main() 