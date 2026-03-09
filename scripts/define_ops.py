import os

OP_LIST = "model_ops_list.txt"
OUTPUT_HEADER = "kernels/kernels.h"

def op_to_define(op_name: str) -> str:
    define_name = op_name.upper()
    return define_name

def main():
    if not os.path.exists(OP_LIST):
        print(f"[ERROR] Op list not found: {OP_LIST}")
        return

    with open(OP_LIST, "r") as f:
        lines = f.readlines()

    defines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        op = parts[0]  

        define_name = op_to_define(op)
        defines.append(define_name)

    os.makedirs(os.path.dirname(OUTPUT_HEADER), exist_ok=True)

    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef TINYGEN_KERNELS_H\n")
        f.write("#define TINYGEN_KERNELS_H\n\n")

        for d in defines:
            f.write(f"#define {d}\n")

        f.write("\n#endif \n")


if __name__ == "__main__":
    main()

