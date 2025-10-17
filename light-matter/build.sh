#!/bin/bash

# Set up the ARPACK library path
export ARPACK_PATH=/vast/home/mike_tay/spack/opt/spack/linux-rhel8-haswell/gcc-9.4.0/arpack-ng-3.9.0-v3yoapcihbvwth4yo35fbbcmpuljes7b
export LD_LIBRARY_PATH=$ARPACK_PATH/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ARPACK_PATH/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=$ARPACK_PATH/include/arpack-ng:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$ARPACK_PATH/include/arpack-ng:$CPLUS_INCLUDE_PATH
export PKG_CONFIG_PATH=$ARPACK_PATH/lib64/pkgconfig:$PKG_CONFIG_PATH

# Run cargo with the updated environment
cargo "$@"
