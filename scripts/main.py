import subprocess
import shutil
import sys
import os
import argparse  

TINYGENC = "./build/compiler/tinyc"
OUTPUT_DIR = "./gen"

def main():
    parser = argparse.ArgumentParser(description="TinyGen Model Compiler Entry Point")
    parser.add_argument("model_path", help="Path to the input .mlir(TOSA) model file (e.g., model.mlir)")
    args = parser.parse_args()

    model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = os.path.basename(model_path)
    model_name = os.path.splitext(filename)[0]
    
    subprocess.run([TINYGENC, model_path], check=True)

    tmp_output = "model_output.mlir"
    emitc_output = os.path.join(OUTPUT_DIR, f"{model_name}_emitc.mlir")
    shutil.move(tmp_output, emitc_output)

    cpp_output = os.path.join(OUTPUT_DIR, f"{model_name}.cpp")
    translate_cmd = ["mlir-translate", "--mlir-to-cpp", emitc_output]
    with open(cpp_output, "w") as f:
        subprocess.run(translate_cmd, stdout=f, check=True)

    subprocess.run([sys.executable, "scripts/define_ops.py"], check=True)

    tinygen_model_name = f"{model_name}_tinygen"
    copy_cmd = [sys.executable, "scripts/copy_kernels.py", "-m", tinygen_model_name]
    subprocess.run(copy_cmd, check=True)

    tinygen_dir = os.path.join(OUTPUT_DIR, tinygen_model_name)
    os.makedirs(tinygen_dir, exist_ok=True)

    tmp_ops = "model_ops_list.txt"
    dst_ops = os.path.join(tinygen_dir, "model_ops_list.txt")
    shutil.move(tmp_ops, dst_ops)

    kernels_optimization_cmd = [
        sys.executable, 
        "scripts/kernels_optimization.py",
        "--dir", tinygen_dir
    ]
    subprocess.run(kernels_optimization_cmd, check=True)

    dst_emitc = os.path.join(tinygen_dir, f"{model_name}_emitc.mlir")
    dst_cpp = os.path.join(tinygen_dir, f"{model_name}.cpp")
    shutil.move(cpp_output, dst_cpp)
    shutil.move(emitc_output, dst_emitc)


if __name__ == "__main__":
    main()
