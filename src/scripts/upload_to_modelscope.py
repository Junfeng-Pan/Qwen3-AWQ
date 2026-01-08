import os
import argparse
import subprocess
import shutil
from modelscope.hub.api import HubApi
from modelscope.hub.errors import NotExistError

def run_command(command, cwd=None, shell=False):
    """Run shell command and stream output."""
    print(f"Executing: {' '.join(command) if isinstance(command, list) else command}")
    
    # Enable LFS progress
    env = os.environ.copy()
    env["GIT_LFS_PROGRESS"] = "true"
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout to capture git progress
        shell=shell,
        cwd=cwd,
        text=True,
        bufsize=1, # Line buffered
        env=env
    )
    
    # Read output char by char or line by line to ensure real-time updates
    # Git progress often uses \r to update lines, so we need to handle that.
    while True:
        # Read a chunk
        output = process.stdout.read(1)
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end="", flush=True)
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

def upload_model(model_dir, model_id, token):
    print(f"=== ModelScope Upload Tool ===")
    print(f"Model Directory: {model_dir}")
    print(f"Model ID: {model_id}")
    
    api = HubApi()
    if token:
        api.login(token)
        
    # 1. Ensure Model Repo Exists
    print("\n[1/5] Checking/Creating Model Repository...")
    try:
        api.get_model(model_id=model_id)
        print(f"Repository {model_id} already exists.")
    except NotExistError:
        print(f"Repository {model_id} does not exist. Creating...")
        try:
            api.create_model(model_id=model_id, visibility=1) # Private by default
            print("Repository created successfully.")
        except Exception as e:
            print(f"Failed to create repository: {e}")
            return
    except Exception as e:
        print(f"Error checking repository: {e}")
        # Proceeding anyway, maybe it exists but get_model failed due to permissions
        
    # 2. Initialize Git
    print("\n[2/5] Initializing Git Repository...")
    git_dir = os.path.join(model_dir, ".git")
    if os.path.exists(git_dir):
        print("Existing .git directory found. Removing to start fresh (safer for upload script)...")
        shutil.rmtree(git_dir)
    
    run_command(["git", "init"], cwd=model_dir)
    run_command(["git", "lfs", "install"], cwd=model_dir)
    
    # Configure User (Local only)
    run_command(["git", "config", "user.name", "ModelScope Uploader"], cwd=model_dir)
    run_command(["git", "config", "user.email", "uploader@example.com"], cwd=model_dir)
    
    # 3. Configure Remote
    # URL Format: https://oauth2:TOKEN@www.modelscope.cn/MODEL_ID.git
    if token:
        remote_url = f"https://oauth2:{token}@www.modelscope.cn/{model_id}.git"
    else:
        # Fallback to simple URL, hoping credential helper works
        remote_url = f"https://www.modelscope.cn/{model_id}.git"
        
    run_command(["git", "remote", "add", "origin", remote_url], cwd=model_dir)
    
    # 4. Track Large Files
    print("\n[3/5] Tracking Large Files (LFS)...")
    # Track common large model files
    extensions = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.model", "*.onnx", "*.pb"]
    for ext in extensions:
        run_command(["git", "lfs", "track", ext], cwd=model_dir)
    run_command(["git", "add", ".gitattributes"], cwd=model_dir)

    # 5. Commit and Push
    print("\n[4/5] Adding and Committing Files...")
    run_command(["git", "add", "."], cwd=model_dir)
    run_command(["git", "commit", "-m", "Upload model via CLI script"], cwd=model_dir)
    
    print("\n[5/5] Pushing to ModelScope (This may take a while, progress will be shown below)...")
    try:
        # -f is used because we re-initialized the repo and want to overwrite remote content or sync it.
        # But if remote is not empty and has different history, -f is needed.
        run_command(["git", "push", "-u", "origin", "master", "-f"], cwd=model_dir)
        print(f"\n✅ Upload Successful! View your model at: https://modelscope.cn/models/{model_id}")
    except subprocess.CalledProcessError:
        print("\n❌ Push failed. Please check the error logs above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to ModelScope with Progress Bar")
    parser.add_argument("--model_dir", type=str, default="Qwen2.5-VL-7B-Instruct-AWQ", help="Path to the model directory")
    parser.add_argument("--model_id", type=str, required=True, help="Target ModelScope Model ID (e.g., your_username/Qwen2.5-VL-7B-Instruct-AWQ)")
    parser.add_argument("--token", type=str, required=True, help="ModelScope Access Token")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Directory {args.model_dir} does not exist.")
        exit(1)
        
    upload_model(args.model_dir, args.model_id, args.token)