import os

path = (Path(__file__).parent / "../../artifacts").resolve()
print("PATH =", path)
print("FILES =", os.listdir(path))
