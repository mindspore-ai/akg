rm -rf build/
rm -rf dist/
cd pybind/
g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` pybind.cpp -o c_expression`python3-config --extension-suffix` \
    -L./../lib/ -L/usr/lib/gcc/aarch64-linux-gnu/7.3.0/ \
    -L $ASCEND_HOME_PATH/lib64/ -L $ASCEND_HOME_PATH/aarch64-linux/lib64/ \
    -Wl,-Bstatic -lc_expression \
    -Wl,-Bdynamic -lstdc++ -lruntime -lprofapi -lascendcl
cp *.so ../python/swft/core/
cd ..
rm -rf /python/swft.egg-info
find ./python -type d -name "__pycache__" -exec rm -rf {} +
scp -r docs/ python/swft/
cp README.md python/swft/
python -m build --wheel
pip install dist/swft-0.0.1-py3-none-any.whl --force-reinstall
rm -rf /python/swft.egg-info
rm -rf build/
rm -rf pybind/*.so
