#!/usr/bin/env python3
"""
Training script untuk PhaseNet Indonesia dengan 99% coverage
Window size: 135 detik (13,500 samples) untuk menangkap 99% data Indonesia

CATATAN: Warning TensorFlow tentang 'is_training' placeholder adalah NORMAL dan bisa diabaikan:
"INVALID_ARGUMENT: You must feed a value for placeholder tensor 'is_training'"
Ini terjadi karena TensorFlow mencoba mengoptimalkan graph computation yang tidak digunakan.
"""

import argparse
import os
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Reduce TensorFlow verbose output while keeping errors and important warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=info+, 2=warning+, 3=error+
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory allocation issues
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable some verbose warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia import DataConfig_Indonesia, DataReader_Indonesia_Train
import numpy as np
import pandas as pd
import json
import datetime

class Config:
    """Configuration class for model parameters"""
    def __init__(self, **kwargs):
        # Set default values
        self.X_shape = [13500, 1, 3]
        self.Y_shape = [13500, 1, 3]
        self.n_channel = 3
        self.n_class = 3
        self.sampling_rate = 100
        self.dt = 0.01
        self.use_batch_norm = True
        self.use_dropout = True
        self.drop_rate = 0.15
        self.optimizer = 'adam'
        self.learning_rate = 0.00003
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

def safe_model_restore(sess, checkpoint_path):
    """
    Safely restore model from checkpoint with compatibility handling
    """
    print(f"Attempting to restore from: {checkpoint_path}")
    
    # Always exclude global_step first since it often causes dtype issues
    all_vars = tf.compat.v1.global_variables()
    vars_no_global_step = [v for v in all_vars if 'global_step' not in v.name.lower()]
    
    try:
        # Method 1: Try restore excluding global_step (most common solution)
        saver = tf.compat.v1.train.Saver(vars_no_global_step)
        saver.restore(sess, checkpoint_path)
        print(f"✅ Restore successful (excluded global_step, loaded {len(vars_no_global_step)} vars)")
        return True
        
    except Exception as e:
        print(f"⚠️  Restore without global_step failed: {str(e)}")
        
        try:
            # Method 2: Manual variable mapping (most robust)
            checkpoint_vars = tf.train.list_variables(checkpoint_path)
            checkpoint_var_dict = dict(checkpoint_vars)
            
            current_vars = tf.compat.v1.global_variables()
            restore_ops = []
            successful_vars = []
            
            for var in current_vars:
                var_name = var.name.split(':')[0]
                
                # Skip global_step and other potentially problematic vars
                if 'global_step' in var_name.lower():
                    continue
                    
                if var_name in checkpoint_var_dict:
                    try:
                        # Load variable value from checkpoint
                        var_value = tf.train.load_variable(checkpoint_path, var_name)
                        
                        # Check if shapes match
                        if var.shape.as_list() == list(var_value.shape):
                            restore_ops.append(var.assign(var_value))
                            successful_vars.append(var_name)
                        else:
                            print(f"   Shape mismatch for {var_name}: {var.shape.as_list()} vs {var_value.shape}")
                    except Exception as ve:
                        print(f"   Failed to load {var_name}: {str(ve)}")
            
            if restore_ops:
                sess.run(restore_ops)
                print(f"✅ Manual restore successful ({len(successful_vars)} variables)")
                print(f"   Sample loaded vars: {successful_vars[:5]}{'...' if len(successful_vars) > 5 else ''}")
                return True
            else:
                print("❌ No variables could be restored")
                return False
                
        except Exception as e2:
            print(f"❌ All restore methods failed: {str(e2)}")
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
    
    # Create datasets
    train_dataset = data_reader_train.dataset(args.batch_size, shuffle=True, drop_remainder=True)
    train_batch = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
    
    if data_reader_valid:
        valid_dataset = data_reader_valid.dataset(args.batch_size, shuffle=False, drop_remainder=False)
        valid_batch = tf.compat.v1.data.make_one_shot_iterator(valid_dataset).get_next()
    else:
        valid_batch = None
    
    # Create model
    model = UNet(config=config, input_batch=train_batch, mode='train')
    
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
                print(f"❌ No checkpoint found in {args.load_model_dir}")
        
        # Create saver for saving checkpoints (always create after loading)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        
        # Training loop
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Window size: {data_reader_train.config.window_length} samples ({data_reader_train.config.window_length/100:.1f}s)")
        
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
            
            # Training phase
            train_losses = []
            step = 0
            
            try:
                while True:
                    feed_dict = {
                        model.is_training: True,
                        model.drop_rate: args.drop_rate
                    }
                    
                    _, loss_val = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                    train_losses.append(loss_val)
                    
                    step += 1
                    if step % 10 == 0:
                        print(f"  Step {step}, Loss: {loss_val:.6f}")
                        
            except tf.errors.OutOfRangeError:
                avg_train_loss = np.mean(train_losses)
                print(f"  Average Training Loss: {avg_train_loss:.6f}")
            
            # Validation phase - recreate dataset for each epoch
            if data_reader_valid:
                print(f"  Running validation...")
                valid_losses = []
                valid_step = 0
                
                # Create fresh validation dataset for this epoch
                valid_dataset_epoch = data_reader_valid.dataset(args.batch_size, shuffle=False, drop_remainder=False)
                valid_batch_epoch = tf.compat.v1.data.make_one_shot_iterator(valid_dataset_epoch).get_next()
                
                try:
                    while True:
                        feed_dict = {
                            model.is_training: False,
                            model.drop_rate: 0.0
                        }
                        
                        # Get validation batch data and run validation
                        sample_batch, target_batch, fname_batch = sess.run(valid_batch_epoch)
                        
                        # Create feed dict with validation data
                        feed_dict[model.X] = sample_batch
                        feed_dict[model.Y] = target_batch
                        
                        loss_val = sess.run(model.loss, feed_dict=feed_dict)
                        valid_losses.append(loss_val)
                        valid_step += 1
                        
                        # Process at most 20 validation batches per epoch
                        if valid_step >= 20:
                            break
                        
                except tf.errors.OutOfRangeError:
                    pass  # End of validation data
                
                if len(valid_losses) > 0:
                    avg_valid_loss = np.mean(valid_losses)
                    print(f"  Average Validation Loss: {avg_valid_loss:.6f} ({len(valid_losses)} batches)")
                else:
                    print(f"  Average Validation Loss: No validation data processed")
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
    parser.add_argument('--learning_rate', type=float, default=0.00003, help='Learning rate (lower for stability)')
    parser.add_argument('--drop_rate', type=float, default=0.15, help='Dropout rate (higher for regularization)')
    parser.add_argument('--decay_step', type=int, default=8, help='Decay step')
    parser.add_argument('--decay_rate', type=float, default=0.92, help='Decay rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='Loss type')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--summary', action='store_true', help='Enable summary')
    
    # Window parameters
    parser.add_argument('--window_length', type=int, default=13500, help='Window length: 13500 samples (135s)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    
    print("=== KONFIGURASI TRAINING INDONESIA 99% COVERAGE ===")
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