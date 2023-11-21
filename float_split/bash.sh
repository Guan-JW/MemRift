python3 setup.py build_ext --inplace    # 就地生成 .so, 不会安装到 site-packages. 针对contiguous tensor的kernel
python3 test.py     # 测试