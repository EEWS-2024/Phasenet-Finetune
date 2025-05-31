#!/usr/bin/env python3
"""
Training script untuk PhaseNet Indonesia dengan Sliding Window 3000 samples
Kompatibel dengan model pretrained 190703-214543 (NCEDC dataset)

CATATAN: Warning TensorFlow tentang 'is_training' placeholder adalah NORMAL dan bisa diabaikan.
"""

import argparse
import os
import sys

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
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
    """Create and save training/validation loss curves plot"""
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    plt.title('Training Loss over Epochs (3000-sample Windows)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot both losses together
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
    
    plt.title('Training vs Validation Loss (Sliding Window Strategy)', fontsize=14, fontweight='bold')
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

def print_loss_summary(train_losses, val_losses):
    """Print summary statistics of training"""
    print(f"\n📊 === TRAINING SUMMARY (3000-sample Windows) ===")
    
    if train_losses:
        print(f"Training Loss:")
        print(f"  Initial: {train_losses[0]:.6f}")
        print(f"  Final: {train_losses[-1]:.6f}")
        print(f"  Best: {min(train_losses):.6f} (epoch {train_losses.index(min(train_losses)) + 1})")
        print(f"  Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
    
    if val_losses:
        print(f"Validation Loss:")
        print(f"  Initial: {val_losses[0]:.6f}")
        print(f"  Final: {val_losses[-1]:.6f}")
        print(f"  Best: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses)) + 1})")
        print(f"  Improvement: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.2f}%")
        
        if len(val_losses) >= 5:
            recent_val = val_losses[-5:]
            if all(recent_val[i] >= recent_val[i-1] for i in range(1, len(recent_val))):
                print(f"  ⚠️  Warning: Validation loss increasing in last 5 epochs (possible overfitting)")
    
    print(f"=" * 40)

def load_pretrained_model(sess, pretrained_model_path):
    """Load pretrained model dengan careful handling untuk stability dan detailed logging"""
    print(f"\n🔄 Loading pretrained model from: {pretrained_model_path}")
    
    try:
        # Check if checkpoint exists
        ckpt = tf.train.latest_checkpoint(pretrained_model_path)
        if not ckpt:
            print(f"❌ No checkpoint found in {pretrained_model_path}")
            return False
        
        print(f"Found checkpoint: {ckpt}")
        
        # Get checkpoint variables
        checkpoint_vars = tf.train.list_variables(ckpt)
        print(f"Checkpoint contains {len(checkpoint_vars)} variables")
        
        # Get current model variables
        current_vars = tf.compat.v1.global_variables()
        print(f"Current model has {len(current_vars)} variables")
        
        # Categorize variables
        loaded_count = 0
        skipped_optimizer = 0
        skipped_shape_mismatch = 0
        skipped_not_found = 0
        skipped_other = 0
        
        # Create checkpoint variable name set untuk faster lookup
        checkpoint_var_names = {name for name, shape in checkpoint_vars}
        
        for var in current_vars:
            var_name = var.name.split(':')[0]
            
            # Skip optimizer variables dan batch norm moving averages
            skip_patterns = ['Adam', 'adam', 'Momentum', 'momentum', 'global_step', 
                           'moving_mean', 'moving_variance', 'beta1_power', 'beta2_power',
                           'learning_rate']  # Add learning_rate variable
            
            if any(pattern in var_name for pattern in skip_patterns):
                skipped_optimizer += 1
                continue
            
            # Check if variable exists in checkpoint
            if var_name not in checkpoint_var_names:
                skipped_not_found += 1
                print(f"  🔍 NOT FOUND: {var_name}")
                continue
            
            try:
                # Try to load this variable
                checkpoint_value = tf.train.load_variable(ckpt, var_name)
                
                # Check if shapes match (should match for 3000-sample model)
                if var.shape.as_list() == list(checkpoint_value.shape):
                    # IMPORTANT: Scale down weights untuk numerical stability
                    if 'kernel' in var_name or 'weight' in var_name:
                        # Scale down conv/dense weights significantly untuk transfer learning stability
                        scaled_value = checkpoint_value * 0.1  # 10x smaller
                    elif 'bias' in var_name:
                        # Scale down bias terms even more
                        scaled_value = checkpoint_value * 0.01  # 100x smaller
                    else:
                        # Other variables (batch norm gamma, etc.)
                        scaled_value = checkpoint_value * 0.5  # 2x smaller
                    
                    sess.run(var.assign(scaled_value))
                    loaded_count += 1
                else:
                    print(f"  ⚠️  SHAPE MISMATCH: {var_name} - Current: {var.shape.as_list()} vs Checkpoint: {list(checkpoint_value.shape)}")
                    skipped_shape_mismatch += 1
                    
            except Exception as e:
                print(f"  ❌ ERROR: {var_name} - {str(e)}")
                skipped_other += 1
        
        total_skipped = skipped_optimizer + skipped_shape_mismatch + skipped_not_found + skipped_other
        
        print(f"\n📊 LOADING SUMMARY:")
        print(f"  ✅ Loaded successfully: {loaded_count}")
        print(f"  ⏭️  Skipped (optimizer/lr): {skipped_optimizer}")
        print(f"  ⚠️  Skipped (not found): {skipped_not_found}")
        print(f"  ⚠️  Skipped (shape mismatch): {skipped_shape_mismatch}")
        print(f"  ❌ Skipped (errors): {skipped_other}")
        print(f"  📈 Total variables: {len(current_vars)}")
        print(f"  📉 Total skipped: {total_skipped}")
        
        return loaded_count > 0 
            
    except Exception as e:
        print(f"❌ Error loading pretrained model: {str(e)}")
        return False

def train_fn(args, data_reader_train, data_reader_valid=None):
    """Training function untuk sliding window 3000 samples"""
    
    # Create model directory with timestamp
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    model_dir = os.path.join(args.model_dir, f"sliding3000_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
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
    
    # Add gradient clipping yang lebih agresif
    learning_rate_var = tf.compat.v1.Variable(config.learning_rate, trainable=False, name='learning_rate')
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_var)
    gradients = tf.gradients(model.loss, tf.compat.v1.trainable_variables())
    
    # Clip gradients very aggressively untuk prevent explosion (penting untuk transfer learning)
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, clip_norm=0.5)  # Much lower clip norm
    
    # Check gradient norms untuk debugging
    gradient_check = tf.debugging.check_numerics(gradient_norm, "Gradient norm contains NaN or Inf")
    
    # Create training operation dengan clipped gradients
    train_op = optimizer.apply_gradients(zip(clipped_gradients, tf.compat.v1.trainable_variables()))
    
    # Override the model's train_op with our gradient-clipped version
    model.train_op = train_op
    
    # Add numerical stability checks
    loss_check = tf.debugging.check_numerics(model.loss, "Loss contains NaN or Inf")
    
    # Add weight checks untuk ensure stability
    weight_checks = []
    for var in tf.compat.v1.trainable_variables():
        weight_checks.append(tf.debugging.check_numerics(var, f"Weight {var.name} contains NaN or Inf"))
    
    # Session config
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False
    sess_config.allow_soft_placement = True
    
    with tf.compat.v1.Session(config=sess_config) as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Pretrained model loading dengan fallback
        pretrained_loaded = False
        if args.pretrained_model_path:
            pretrained_loaded = load_pretrained_model(sess, args.pretrained_model_path)
        
        if not pretrained_loaded:
            print("⚠️  Fallback to Xavier initialization untuk stability")
            # Initialize weights dengan Xavier (lebih stable untuk training dari awal)
            for var in tf.compat.v1.global_variables():
                if 'kernel' in var.name or 'weight' in var.name:
                    # Xavier uniform initialization untuk convolution weights
                    fan_in = np.prod(var.shape.as_list()[:-1])
                    fan_out = var.shape.as_list()[-1]
                    bound = np.sqrt(6.0 / (fan_in + fan_out))
                    sess.run(var.assign(tf.random.uniform(var.shape, -bound, bound)))
                elif 'bias' in var.name:
                    # Initialize bias to zero
                    sess.run(var.assign(tf.zeros(var.shape)))
        
        # Warmup learning rate untuk first few epochs (gradual scaling)
        base_lr = config.learning_rate
        print(f"🔥 Starting training dengan warmup period...")
        print(f"   Base learning rate: {base_lr}")
        print(f"   Warmup akan digunakan untuk 3 epochs pertama")
        
        # Initialize loss tracking
        training_losses = []
        training_epochs = []
        
        print(f"\n🚀 Starting training for {config.epochs} epochs...")
        print(f"Sliding window: {data_reader_train.window_length} samples ({data_reader_train.window_length/100:.1f}s)")
        print(f"Total training windows: {len(data_reader_train.sliding_windows)}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        
        # Create saver
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        
        # Create dataset iterators
        train_dataset = data_reader_train.dataset(config.batch_size, shuffle=True, drop_remainder=True)
        train_iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
        next_train_batch = train_iterator.get_next()
        
        # Training loop dengan warmup dan safety checks
        for epoch in range(config.epochs):
            print(f"\n📊 Epoch {epoch+1}/{config.epochs}")
            
            # Adjust learning rate untuk warmup atau decay
            if epoch < 3:
                warmup_factor = 0.1 + 0.9 * (epoch / 3.0)  # 0.1x to 1.0x gradually
                current_lr = base_lr * warmup_factor
                print(f"   🔥 Warmup mode: LR = {current_lr:.8f} ({warmup_factor:.1%} of base)")
            else:
                # Normal learning rate with decay
                current_lr = base_lr * (config.decay_rate ** ((epoch-3) // config.decay_step))
                print(f"   📉 Normal mode: LR = {current_lr:.8f}")
            
            # Update learning rate in optimizer
            sess.run(tf.compat.v1.assign(learning_rate_var, current_lr))
            
            # Reset epoch metrics dan dataset iterator
            epoch_losses = []
            sess.run(train_iterator.initializer)
            
            # Calculate number of steps per epoch
            training_steps = int(np.ceil(len(data_reader_train.sliding_windows) / config.batch_size))
            
            # Training loop untuk epoch ini
            step = 0
            try:
                with tqdm(total=training_steps, desc=f"Epoch {epoch+1}/{config.epochs}") as pbar:
                    while True:
                        try:
                            # Get next batch dari dataset iterator
                            X_batch, Y_batch, fname_batch = sess.run(next_train_batch)
                            
                            # Safety check input data
                            if np.any(np.isnan(X_batch)) or np.any(np.isinf(X_batch)):
                                pbar.write(f"⚠️  NaN or Inf detected in input batch, skipping...")
                                continue
                            
                            if np.any(np.isnan(Y_batch)) or np.any(np.isinf(Y_batch)):
                                pbar.write(f"⚠️  NaN or Inf detected in target batch, skipping...")
                                continue
                            
                            # Run training step dengan checks
                            feed_dict = {
                                X_placeholder: X_batch,
                                Y_placeholder: Y_batch,
                                fname_placeholder: fname_batch,
                                model.is_training: True,
                                model.drop_rate: config.drop_rate
                            }
                            
                            # Run training step
                            _, step_loss, step_logits = sess.run([model.train_op, model.loss, model.logits], feed_dict=feed_dict)
                            
                            # Safety check output
                            if np.isnan(step_loss) or np.isinf(step_loss):
                                pbar.write(f"❌ NaN/Inf loss detected at step {step}!")
                                pbar.write(f"   Learning rate: {current_lr}")
                                pbar.write(f"   Reducing learning rate by 10x and retrying...")
                                
                                # Reduce learning rate dramatically
                                base_lr = base_lr * 0.1
                                current_lr = base_lr * warmup_factor if epoch < 3 else base_lr * (config.decay_rate ** ((epoch-3) // config.decay_step))
                                sess.run(tf.compat.v1.assign(learning_rate_var, current_lr))
                                
                                # Skip this batch
                                continue
                            
                            epoch_losses.append(step_loss)
                            
                            # Print progress every 100 steps (reduced frequency)
                            if step % 100 == 0:
                                pbar.write(f"   Step {step}: Loss = {step_loss:.6f}, LR = {current_lr:.8f}")
                            
                            step += 1
                            pbar.update(1)
                        
                        except tf.errors.OutOfRangeError:
                            # End of dataset reached
                            break
                        except Exception as e:
                            pbar.write(f"⚠️  Error in training step {step}: {str(e)}")
                            step += 1
                            pbar.update(1)
                            continue
                            
            except KeyboardInterrupt:
                print(f"\n⚠️  Training interrupted by user")
                break
            
            # Epoch summary
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"✅ Epoch {epoch+1} completed: Average Loss = {avg_loss:.6f}")
                
                # Record training metrics
                training_losses.append(avg_loss)
                training_epochs.append(epoch + 1)
                
                # Early stopping check jika loss explode
                if avg_loss > 10.0:  # Loss too high
                    print(f"⚠️  Loss too high ({avg_loss:.6f}), reducing learning rate by 5x")
                    base_lr = base_lr * 0.2
            else:
                print(f"⚠️  No valid losses recorded for epoch {epoch+1}")
                training_losses.append(float('inf'))
                training_epochs.append(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % config.save_interval == 0 or epoch == config.epochs - 1:
                checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.ckpt")
                saver.save(sess, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.ckpt")
        saver.save(sess, final_model_path)
        print(f"\nFinal model saved: {final_model_path}")
        
        # Save loss history and create plots
        print(f"\n📊 Saving training history and plots...")
        
        # Clean NaN values for plotting
        clean_train_losses = [x for x in training_losses if not np.isnan(x)]
        
        # Save CSV file
        save_loss_history(training_losses, [], model_dir)
        
        # Create and save plots
        if clean_train_losses:
            plot_loss_curves(clean_train_losses, [], model_dir)
        
        # Print training summary
        print_loss_summary(clean_train_losses, [])
        
        return model_dir

def main():
    parser = argparse.ArgumentParser(description='Training PhaseNet Indonesia dengan Sliding Window 3000 samples')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--train_list', type=str, required=True, help='Training data list CSV')
    parser.add_argument('--valid_dir', type=str, help='Validation data directory')
    parser.add_argument('--valid_list', type=str, help='Validation data list CSV')
    parser.add_argument('--format', type=str, default='numpy', help='Data format')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, default='model_indonesia/finetuned', help='Model directory')
    parser.add_argument('--pretrained_model_path', type=str, default='model/190703-214543', help='Path to pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs_indonesia', help='Log directory')
    
    # Training parameters optimized for sliding window
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (fewer needed with pretrained)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (can be larger with 3000 samples)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate (higher for fine-tuning)')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--decay_step', type=int, default=5, help='Decay step')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--summary', action='store_true', help='Enable summary')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    
    print("=== KONFIGURASI TRAINING INDONESIA 3000-SAMPLE SLIDING WINDOW ===")
    print(f"Window length: 3000 samples (30.0 detik)")
    print(f"Training directory: {args.train_dir}")
    print(f"Training list: {args.train_list}")
    print(f"Model directory: {args.model_dir}")
    print(f"Pretrained model: {args.pretrained_model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Create data config
    data_config = DataConfig_Indonesia_3000(
        window_length=3000,
        X_shape=[3000, 1, 3],
        Y_shape=[3000, 1, 3]
    )
    
    # Create training data reader
    data_reader_train = DataReader_Indonesia_Sliding_Train(
        format=args.format,
        config=data_config,
        data_dir=args.train_dir,
        data_list=args.train_list
    )
    
    # Create validation data reader if provided
    data_reader_valid = None
    if args.valid_dir and args.valid_list:
        data_reader_valid = DataReader_Indonesia_Sliding_Train(
            format=args.format,
            config=data_config,
            data_dir=args.valid_dir,
            data_list=args.valid_list
        )
    
    # Start training
    model_dir = train_fn(args, data_reader_train, data_reader_valid)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved in: {model_dir}")
    print(f"\n🎯 Model trained dengan sliding window strategy dan pretrained weights!")

if __name__ == '__main__':
    main() 