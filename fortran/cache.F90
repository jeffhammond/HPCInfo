module cache

    ! in practice, the compiler should get these from the OS
    ! e.g. https://github.com/torvalds/linux/blob/06dc10eae55b5ceabfef287a7e5f16ceea204aa0/arch/sh/include/asm/cache.h#L15
    ! but this is a toy example

    integer, parameter :: line_size = &
#if defined(x86_64) || defined(x86) || defined(aarch64) || defined(arm32)
        64
#elif defined(ppc64le) || defined(ppc64) || defined(ppc32)
        128
#elif defined(NVIDIA)
        ! it's complicated...
        ! https://forums.developer.nvidia.com/t/cache-line-size-of-l1-and-l2/24907
        32
#else
        64
#endif

end module cache

