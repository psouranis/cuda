### Tiled convolution using caches for halo cells

In the `conv2d_sm.cu` much of the complexity has to do with the fact that the input tiles and blocks are larger than the output tiles because of the loading of halo cells. 
But each halo cell of an input tile of a block are also the internal elements of neighboring tiles of the input tiles of neighboring blocks.

That tells us that there is a significant probability that by the time a block needs its halo cells, they are already in L2 cache because of the accesses by its neighboring blocks. As a result, the memory accesses to these halo cells may be naturally served from L2 cache without using additional DRAM traffic. 

That is, we can leave the accesses to these halo cells in the original N elements rather than loading them into the `N_ds`.