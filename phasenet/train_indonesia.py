#!/usr/bin/env python3
"""
Training script untuk PhaseNet Indonesia

CATATAN: Warning TensorFlow tentang 'is_training' placeholder adalah NORMAL dan bisa diabaikan:
"INVALID_ARGUMENT: You must feed a value for placeholder tensor 'is_training'"
Ini terjadi karena TensorFlow mencoba mengoptimalkan graph computation yang tidak digunakan.
"""

import argparse
import os
import sys

# Set environment variables BEFORE importing TensorFlow to prevent XLA issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=info+, 2=warning+, 3=error+
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory allocation issues
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Disable XLA to avoid libdevice issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable some verbose warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend untuk server

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia import DataConfig_Indonesia, DataReader_Indonesia_Train
import json
import datetime

class Config:
    """Configuration class for model parameters"""
    def __init__(self, **kwargs):
        # Set default values untuk 170s window dengan 10s buffers
        self.X_shape = [17000, 1, 3]
        self.Y_shape = [17000, 1, 3]
        self.n_channel = 3
        self.n_class = 3
        self.sampling_rate = 100
        self.dt = 0.01
        self.use_batch_norm = True
        self.use_dropout = True
        self.drop_rate = 0.15
        self.optimizer = 'adam'
        self.learning_rate = 0.000015
        self.decay_step = 8
        self.decay_rate = 0.92
        self.batch_size = 16
        self.epochs = 100
        self.loss_type = 'cross_entropy'
        self.weight_decay = 0.0001
        self.summary = True
        self.class_weights = [1.0, 1.0, 1.0]
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

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
    # Create DataFrame with loss history
    max_epochs = max(len(train_losses), len(val_losses))
    
    # Pad shorter list with None
    train_padded = train_losses + [None] * (max_epochs - len(train_losses))
    val_padded = val_losses + [None] * (max_epochs - len(val_losses))
    
    loss_df = pd.DataFrame({
        'epoch': range(1, max_epochs + 1),
        'train_loss': train_padded,
        'val_loss': val_padded
    })
    
    # Save to CSV
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
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot both losses together if validation exists
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
    
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss curves plot saved to: {plot_path}")
    
    # Also save as PDF for high quality
    plot_path_pdf = os.path.join(model_dir, 'loss_curves.pdf')
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
    
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss curves plot (PDF) saved to: {plot_path_pdf}")

def print_loss_summary(train_losses, val_losses):
    """Print summary statistics of training"""
    print(f"\n📊 === TRAINING SUMMARY ===")
    
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
        
        # Check for overfitting
        if len(val_losses) >= 5:
            recent_val = val_losses[-5:]
            if all(recent_val[i] >= recent_val[i-1] for i in range(1, len(recent_val))):
                print(f"  ⚠️  Warning: Validation loss increasing in last 5 epochs (possible overfitting)")
    
    print(f"=" * 40)

def safe_model_restore(sess, checkpoint_path):
    """
    Smart transfer learning: load only compatible weights from 30s model to 170s model
    This is specifically designed for 190703-214543 (30s) -> Indonesia 170s transfer
    """
    print(f"🧠 Smart Transfer Learning from: {checkpoint_path}")
    print("   Strategy: Load compatible conv layers, reinit incompatible layers")
    
    try:
        # Get checkpoint variables
        checkpoint_vars = tf.train.list_variables(checkpoint_path)
        checkpoint_var_dict = dict(checkpoint_vars)
        print(f"   Found {len(checkpoint_vars)} variables in checkpoint")
        
        # Get current model variables
        current_vars = tf.compat.v1.global_variables()
        
        # Categories for smart loading
        loaded_vars = []
        skipped_vars = []
        reinit_vars = []
        
        for var in current_vars:
            var_name = var.name.split(':')[0]
            
            # ALWAYS skip these problematic variables
            skip_patterns = [
                'global_step', 'Adam', 'adam', 'Momentum', 'momentum',
                'moving_mean', 'moving_variance', 'beta', 'gamma'
            ]
            
            if any(pattern in var_name for pattern in skip_patterns):
                skipped_vars.append(var_name)
                continue
            
            # Check if variable exists in checkpoint
            if var_name not in checkpoint_var_dict:
                reinit_vars.append(f"{var_name} (not_in_checkpoint)")
                continue
            
            try:
                # Load checkpoint value
                checkpoint_value = tf.train.load_variable(checkpoint_path, var_name)
                current_shape = var.shape.as_list()
                checkpoint_shape = list(checkpoint_value.shape)
                
                # STRATEGY 1: Perfect match - load directly
                if current_shape == checkpoint_shape:
                    # Scale down weights for stability in transfer learning
                    scaled_value = checkpoint_value * 0.1  # 10x smaller for stability
                    sess.run(var.assign(scaled_value))
                    loaded_vars.append(f"{var_name} {current_shape} (scaled_0.1x)")
                    
                # STRATEGY 2: Conv layers - load if channels match
                elif len(current_shape) == 4 and len(checkpoint_shape) == 4:
                    # Conv weight: [height, width, in_channels, out_channels]
                    if (current_shape[2] == checkpoint_shape[2] and 
                        current_shape[3] == checkpoint_shape[3]):
                        # Same in/out channels - can initialize with small kernel
                        if current_shape[0] <= checkpoint_shape[0] and current_shape[1] <= checkpoint_shape[1]:
                            # Current kernel smaller or equal - crop checkpoint kernel
                            h_start = (checkpoint_shape[0] - current_shape[0]) // 2
                            w_start = (checkpoint_shape[1] - current_shape[1]) // 2
                            cropped_value = checkpoint_value[
                                h_start:h_start+current_shape[0], 
                                w_start:w_start+current_shape[1], 
                                :, :
                            ]
                            # Scale down for stability
                            scaled_cropped = cropped_value * 0.1
                            sess.run(var.assign(scaled_cropped))
                            loaded_vars.append(f"{var_name} {current_shape} (cropped_scaled)")
                        else:
                            # Current kernel larger - pad with small random values
                            pad_h = (current_shape[0] - checkpoint_shape[0]) // 2
                            pad_w = (current_shape[1] - checkpoint_shape[1]) // 2
                            padded_value = np.pad(
                                checkpoint_value, 
                                ((pad_h, current_shape[0]-checkpoint_shape[0]-pad_h),
                                 (pad_w, current_shape[1]-checkpoint_shape[1]-pad_w),
                                 (0, 0), (0, 0)),
                                mode='constant', constant_values=0
                            )
                            # Add small noise to padded regions and scale down
                            mask = np.ones_like(padded_value)
                            mask[pad_h:pad_h+checkpoint_shape[0], pad_w:pad_w+checkpoint_shape[1], :, :] = 0
                            noise = np.random.normal(0, 0.001, padded_value.shape) * mask
                            final_value = (padded_value + noise) * 0.1  # Scale down for stability
                            sess.run(var.assign(final_value))
                            loaded_vars.append(f"{var_name} {current_shape} (padded_scaled)")
                    else:
                        # Different channels - reinitialize with smart scaling
                        if 'conv' in var_name.lower():
                            # Initialize conv layers with smaller std for stability
                            fan_in = current_shape[0] * current_shape[1] * current_shape[2]
                            std = np.sqrt(2.0 / fan_in)  # He initialization
                            init_value = np.random.normal(0, std * 0.01, current_shape)  # 100x smaller for extreme stability
                            sess.run(var.assign(init_value))
                            reinit_vars.append(f"{var_name} {current_shape} (ultra_small_conv_init)")
                        else:
                            reinit_vars.append(f"{var_name} {current_shape} (channel_mismatch)")
                            
                # STRATEGY 3: Dense/FC layers - very conservative
                elif len(current_shape) == 2 and len(checkpoint_shape) == 2:
                    # Dense layer: [input_size, output_size]
                    if current_shape[1] == checkpoint_shape[1]:  # Same output size
                        if current_shape[0] <= checkpoint_shape[0]:
                            # Take subset of input weights and scale down
                            subset_value = checkpoint_value[:current_shape[0], :] * 0.05  # 20x smaller for FC layers
                            sess.run(var.assign(subset_value))
                            loaded_vars.append(f"{var_name} {current_shape} (subset_scaled)")
                        else:
                            # Pad input weights - but this is risky for very different sizes
                            reinit_vars.append(f"{var_name} {current_shape} (input_size_too_large)")
                    else:
                        # Different output size - must reinitialize
                        reinit_vars.append(f"{var_name} {current_shape} (output_mismatch)")
                        
                # STRATEGY 4: Bias vectors
                elif len(current_shape) == 1 and len(checkpoint_shape) == 1:
                    if current_shape[0] == checkpoint_shape[0]:
                        # Scale down bias terms significantly for stability
                        scaled_bias = checkpoint_value * 0.01  # 100x smaller for bias
                        sess.run(var.assign(scaled_bias))
                        loaded_vars.append(f"{var_name} {current_shape} (bias_scaled)")
                    else:
                        reinit_vars.append(f"{var_name} {current_shape} (bias_size_mismatch)")
                        
                else:
                    # Unsupported shape combination
                    reinit_vars.append(f"{var_name} {current_shape} vs {checkpoint_shape} (unsupported)")
                    
            except Exception as e:
                reinit_vars.append(f"{var_name} (load_error: {str(e)[:50]})")
                
        # Report results
        print(f"📊 Smart Transfer Learning Results:")
        print(f"   ✅ Loaded: {len(loaded_vars)} variables")
        print(f"   ⏭️  Skipped: {len(skipped_vars)} variables (optimizer/batch_norm)")
        print(f"   🎲 Reinit: {len(reinit_vars)} variables (incompatible)")
        
        if len(loaded_vars) > 0:
            print(f"   💾 Sample loaded: {loaded_vars[:3]}")
        if len(reinit_vars) > 0:
            print(f"   🎲 Sample reinit: {reinit_vars[:3]}")
            
        # Success if we loaded at least some conv layers
        conv_loaded = sum(1 for v in loaded_vars if 'conv' in v.lower())
        if conv_loaded > 0:
            print(f"   🎯 Transfer learning SUCCESS: {conv_loaded} conv layers loaded")
            return True
        else:
            print(f"   ⚠️  WARNING: No conv layers loaded, this is essentially random init")
            return False
            
    except Exception as e:
        print(f"❌ Smart transfer learning failed: {str(e)}")
        return False

def train_fn(args, data_reader_train, data_reader_valid=None):
    """Training function for Indonesia 99% coverage model"""
    
    # Create model directory with timestamp
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    model_dir = os.path.join(args.model_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
    # Create model config
    config = ModelConfig(
        X_shape=data_reader_train.config.X_shape,
        Y_shape=data_reader_train.config.Y_shape,
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
        class_weights=[1.0, 1.0, 1.0]
    )
    
    # Save configuration
    save_config(config, model_dir)
    
    # Create datasets with proper re-initialization
    train_dataset = data_reader_train.dataset(args.batch_size, shuffle=True, drop_remainder=True)
    
    # Use placeholder and feed_dict approach for more stable training
    X_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + data_reader_train.config.X_shape, name='X_input')
    Y_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + data_reader_train.config.Y_shape, name='Y_target')
    fname_placeholder = tf.compat.v1.placeholder(tf.string, [None], name='fname_input')
    
    # Create model with placeholders
    model = UNet(config=config, input_batch=(X_placeholder, Y_placeholder, fname_placeholder), mode='train')
    
    # Add gradient clipping to prevent gradient explosion and NaN losses
    optimizer = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
    gradients = tf.gradients(model.loss, tf.compat.v1.trainable_variables())
    
    # Clip gradients to prevent explosion (very important for GPU training)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    
    # Create training operation with clipped gradients
    train_op = optimizer.apply_gradients(zip(clipped_gradients, tf.compat.v1.trainable_variables()))
    
    # Override the model's train_op with our gradient-clipped version
    model.train_op = train_op
    
    # Add numerical stability checks
    loss_check = tf.debugging.check_numerics(model.loss, "Loss contains NaN or Inf")
    
    # Session config
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False
    sess_config.allow_soft_placement = True
    
    with tf.compat.v1.Session(config=sess_config) as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Load pre-trained model if specified
        if args.load_model and args.load_model_dir:
            print(f"Loading pre-trained model from: {args.load_model_dir}")
            
            # Find latest checkpoint
            ckpt = tf.train.latest_checkpoint(args.load_model_dir)
            if ckpt:
                success = safe_model_restore(sess, ckpt)
                if not success:
                    print("❌ Failed to load pre-trained model with all methods")
                    print("   Training will continue from scratch")
                else:
                    print("✅ Transfer learning weights loaded successfully")
                    
                    # Test with a small forward pass to check for NaN issues
                    print("🔍 Testing transfer learning stability...")
                    try:
                        # Create a small test batch
                        test_X = tf.random.normal([1] + data_reader_train.config.X_shape)
                        test_Y = tf.random.normal([1] + data_reader_train.config.Y_shape)
                        test_fname = tf.constant(["test"])
                        
                        test_feed_dict = {
                            X_placeholder: test_X.eval(),
                            Y_placeholder: test_Y.eval(),
                            fname_placeholder: test_fname.eval(),
                            model.is_training: False,
                            model.drop_rate: 0.0
                        }
                        
                        test_loss = sess.run(model.loss, feed_dict=test_feed_dict)
                        
                        if np.isnan(test_loss) or np.isinf(test_loss):
                            print(f"⚠️  Transfer learning produces NaN/Inf: {test_loss}")
                            print("   Reinitializing all variables to prevent training issues")
                            sess.run(tf.compat.v1.global_variables_initializer())
                        else:
                            print(f"✅ Transfer learning test passed: loss = {test_loss:.6f}")
                            
                    except Exception as e:
                        print(f"⚠️  Transfer learning test failed: {str(e)}")
                        print("   Reinitializing all variables to prevent training issues")
                        sess.run(tf.compat.v1.global_variables_initializer())
            else:
                print(f"❌ No checkpoint found in {args.load_model_dir}")
        
        # Create saver for saving checkpoints (always create after loading)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        
        # Training loop
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Window size: {data_reader_train.config.window_length} samples ({data_reader_train.config.window_length/100:.1f}s)")
        
        # Initialize loss tracking
        epoch_train_losses = []
        epoch_val_losses = []
        
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
            
            # Create fresh iterator for each epoch
            train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
            train_batch = train_iterator.get_next()
            
            # Training phase
            train_losses = []
            step = 0
            
            try:
                while True:
                    # Get batch data
                    sample_batch, target_batch, fname_batch = sess.run(train_batch)
                    
                    # Create feed dict
                    feed_dict = {
                        X_placeholder: sample_batch,
                        Y_placeholder: target_batch,
                        fname_placeholder: fname_batch,
                        model.is_training: True,
                        model.drop_rate: args.drop_rate
                    }
                    
                    # Run training step with numerical checks
                    try:
                        _, loss_val, _ = sess.run([model.train_op, model.loss, loss_check], feed_dict=feed_dict)
                        
                        # Check for NaN/Inf
                        if np.isnan(loss_val) or np.isinf(loss_val):
                            print(f"⚠️  WARNING: Invalid loss detected: {loss_val}")
                            print(f"   Step: {step}, Epoch: {epoch + 1}")
                            # Skip this batch but continue training
                            continue
                            
                        train_losses.append(loss_val)
                        step += 1
                        
                        if step % 10 == 0:
                            print(f"  Step {step}, Loss: {loss_val:.6f}")
                            
                    except tf.errors.InvalidArgumentError as e:
                        print(f"⚠️  Numerical error detected at step {step}: {str(e)}")
                        print(f"   Skipping this batch and continuing...")
                        continue
                        
            except tf.errors.OutOfRangeError:
                if len(train_losses) > 0:
                    avg_train_loss = np.mean(train_losses)
                    epoch_train_losses.append(avg_train_loss)
                    print(f"  Average Training Loss: {avg_train_loss:.6f}")
                else:
                    epoch_train_losses.append(float('nan'))
                    print(f"  Average Training Loss: No valid batches processed")
            
            # Validation phase with fresh data
            if data_reader_valid:
                print(f"  Running validation...")
                valid_losses = []
                valid_step = 0
                
                # Create fresh validation dataset for this epoch
                valid_dataset_epoch = data_reader_valid.dataset(args.batch_size, shuffle=False, drop_remainder=False)
                valid_iterator = tf.compat.v1.data.make_one_shot_iterator(valid_dataset_epoch)
                valid_batch_epoch = valid_iterator.get_next()
                
                try:
                    while True:
                        # Get validation batch data
                        sample_batch, target_batch, fname_batch = sess.run(valid_batch_epoch)
                        
                        # Create feed dict for validation
                        feed_dict = {
                            X_placeholder: sample_batch,
                            Y_placeholder: target_batch,
                            fname_placeholder: fname_batch,
                            model.is_training: False,
                            model.drop_rate: 0.0
                        }
                        
                        loss_val = sess.run(model.loss, feed_dict=feed_dict)
                        
                        # Check for valid loss
                        if not (np.isnan(loss_val) or np.isinf(loss_val)):
                            valid_losses.append(loss_val)
                        
                        valid_step += 1
                        
                        # Process at most 20 validation batches per epoch
                        if valid_step >= 20:
                            break
                        
                except tf.errors.OutOfRangeError:
                    pass  # End of validation data
                
                if len(valid_losses) > 0:
                    avg_valid_loss = np.mean(valid_losses)
                    epoch_val_losses.append(avg_valid_loss)
                    print(f"  Average Validation Loss: {avg_valid_loss:.6f} ({len(valid_losses)} batches)")
                else:
                    epoch_val_losses.append(float('nan'))
                    print(f"  Average Validation Loss: No valid validation batches")
            else:
                print(f"  No validation data provided")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.ckpt")
                saver.save(sess, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.ckpt")
        saver.save(sess, final_model_path)
        print(f"\nFinal model saved: {final_model_path}")
        
        # Save loss history and create plots
        print(f"\n📊 Saving training history and plots...")
        
        # Clean NaN values for plotting
        clean_train_losses = [x for x in epoch_train_losses if not np.isnan(x)]
        clean_val_losses = [x for x in epoch_val_losses if not np.isnan(x)]
        
        # Save CSV file
        save_loss_history(epoch_train_losses, epoch_val_losses, model_dir)
        
        # Create and save plots
        if clean_train_losses:
            plot_loss_curves(clean_train_losses, clean_val_losses, model_dir)
        
        # Print training summary
        print_loss_summary(clean_train_losses, clean_val_losses)
        
        return model_dir

def main():
    parser = argparse.ArgumentParser(description='Training PhaseNet Indonesia 99% Coverage')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--train_list', type=str, required=True, help='Training data list CSV')
    parser.add_argument('--valid_dir', type=str, help='Validation data directory')
    parser.add_argument('--valid_list', type=str, help='Validation data list CSV')
    parser.add_argument('--format', type=str, default='numpy', help='Data format')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, default='model_indonesia', help='Model directory')
    parser.add_argument('--load_model', action='store_true', help='Load existing model')
    parser.add_argument('--load_model_dir', type=str, help='Directory to load model from')
    parser.add_argument('--log_dir', type=str, default='logs_indonesia', help='Log directory')
    
    # Training parameters optimized for 99% coverage
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (reduced for very large windows)')
    parser.add_argument('--learning_rate', type=float, default=0.000015, help='Learning rate (lower for stability)')
    parser.add_argument('--drop_rate', type=float, default=0.15, help='Dropout rate (higher for regularization)')
    parser.add_argument('--decay_step', type=int, default=8, help='Decay step')
    parser.add_argument('--decay_rate', type=float, default=0.92, help='Decay rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--summary', action='store_true', help='Enable summary')
    
    # Window parameters
    parser.add_argument('--window_length', type=int, default=17000, help='Window length: 17000 samples (170s)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    
    print("=== KONFIGURASI TRAINING INDONESIA===")
    print(f"Window length: {args.window_length} samples ({args.window_length/100:.1f} detik)")
    print(f"Training directory: {args.train_dir}")
    print(f"Training list: {args.train_list}")
    print(f"Model directory: {args.model_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Create data config
    data_config = DataConfig_Indonesia(
        window_length=args.window_length,
        X_shape=[args.window_length, 1, 3],
        Y_shape=[args.window_length, 1, 3]
    )
    
    # Create training data reader
    data_reader_train = DataReader_Indonesia_Train(
        format=args.format,
        config=data_config,
        data_dir=args.train_dir,
        data_list=args.train_list
    )
    
    # Create validation data reader if provided
    data_reader_valid = None
    if args.valid_dir and args.valid_list:
        data_reader_valid = DataReader_Indonesia_Train(
            format=args.format,
            config=data_config,
            data_dir=args.valid_dir,
            data_list=args.valid_list
        )
    
    # Start training
    model_dir = train_fn(args, data_reader_train, data_reader_valid)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved in: {model_dir}")

if __name__ == '__main__':
    main() 