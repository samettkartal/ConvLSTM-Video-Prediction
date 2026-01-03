import zipfile
import os

def create_zip():
    target_files = [
        'config.py',
        'data_loader.py',
        'evaluate.py',
        'layers.py',
        'losses.py',
        'model.py',
        'train.py',
        'utils.py',
        'requirements.txt',
        'generate_video.py',
        'visualize_prediction.py' # Added new script
    ]
    
    target_dirs = [
        'checkpoints' # Added checkpoints directory
    ]
    
    zip_name = 'colab_project.zip'
    
    with zipfile.ZipFile(zip_name, 'w') as zf:
        # Add individual files
        for file in target_files:
            if os.path.exists(file):
                print(f"Adding file: {file}")
                zf.write(file)
            else:
                print(f"Warning: {file} not found!")
        
        # Add directories recursively
        for directory in target_dirs:
            if os.path.exists(directory):
                print(f"Adding directory: {directory}")
                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Archive name should be relative to project root
                        # e.g. 'checkpoints/model_best.keras'
                        # root is like '.../checkpoints' or 'checkpoints'
                        # arcname = os.path.join(root, file) is risky if root is absolute
                        # We assume this script runs in project root.
                        zf.write(file_path, arcname=file_path)
            else:
                print(f"Warning: Directory {directory} not found!")
                
    print(f"Created {zip_name} successfully.")

if __name__ == "__main__":
    create_zip()
