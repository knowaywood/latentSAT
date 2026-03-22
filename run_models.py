import subprocess

scripts = [
    r"src\latentsat\train\list_struct\pretrain.py",
    r"src\latentsat\train\list_struct\rl.py",
    r"src\latentsat\evaluate\eval.py",
]

for script in scripts:
    print(f"开始运行: {script} ...")
    subprocess.run(["uv", "run", script], check=True)
    print(f"{script} 运行完成！\n")
    print("=" * 80)

print("所有文件均已成功运行完毕！")
