# Vitis Accel Examples' Repository
================================

## Running the helloworld example

This code currently contains an example of matirx multiple executed on the U200 FPGA on Pitt CRC. 
Eventually this code will be used as a tool to aid in development of an accelerated matrix search of the Munkres data association algorithm.

ssh to viz node in CRC
Execute the following command: faketime -f '-4y' make host TARGET=hw DEVICE=xilinx_u200_xdma_201830_2 
ssh to fpga-n0 
Execute the following command: faketime -f '-4y' make check TARGET=hw DEVICE=xilinx_u200_xdma_201830_2
- This command should take some time.

The resulting details should be found within the builddir and contain information about resource utilization etc.

This example code was originally derived from the Vitis Accel Examples Repository.