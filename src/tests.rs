// thanks claude, idk what tests to poke at rn.
#[cfg(test)]
mod tests {
    use crate::*;
    use std::thread;

    fn init() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            crate::sgpu_init(&SgpuInititizationInfo::default());
        });
    }

    // Basic sanity — already works, good baseline
    #[test]
    fn test_fill_single_thread() {
        init();
        let buffer = crate::create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: 4 * 100,
            memory_type: MemoryType::HostVisible,
        });

        let mut cmd = crate::record(QueueType::Graphics);
        cmd.fill_buffer(&buffer, 0, 400, 0xDEADBEEF);
        crate::wait(crate::submit(&[cmd]));

        let data: &[u32] = buffer.as_slice();
        assert!(data.iter().all(|&x| x == 0xDEADBEEF));

        crate::destroy_buffer(buffer);
    }

    // Pool reuse — submit many times on same thread, pools should cycle without growing unbounded
    #[test]
    fn test_pool_reuse() {
        init();
        let buffer = crate::create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: 4 * 64,
            memory_type: MemoryType::HostVisible,
        });

        for i in 0..32u32 {
            let mut cmd = crate::record(QueueType::Graphics);
            cmd.fill_buffer(&buffer, 0, 4 * 64, i);
            crate::wait(crate::submit(&[cmd]));

            let data: &[u32] = buffer.as_slice();
            assert!(data.iter().all(|&x| x == i), "iteration {i} failed");
        }

        crate::destroy_buffer(buffer);
    }

    // Multiple threads each get their own ThreadCommandPools,
    // all writing to separate buffers concurrently — no cross-thread pool sharing
    #[test]
    fn test_multithreaded_independent() {
        init();

        const THREAD_COUNT: usize = 4;
        const BUF_SIZE: u64 = 4 * 64;

        let handles: Vec<_> = (0..THREAD_COUNT)
            .map(|i| {
                thread::spawn(move || {
                    let buffer = crate::create_buffer(&BufferDescription {
                        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                        size: BUF_SIZE,
                        memory_type: MemoryType::HostVisible,
                    });

                    // Each thread does several submits to stress pool reuse per thread
                    for j in 0..8u32 {
                        let val = (i as u32) * 100 + j;
                        let mut cmd = crate::record(QueueType::Graphics);
                        cmd.fill_buffer(&buffer, 0, BUF_SIZE, val);
                        crate::wait(crate::submit(&[cmd]));

                        let data: &[u32] = buffer.as_slice();
                        assert!(data.iter().all(|&x| x == val), "thread {i} iter {j} failed");
                    }

                    crate::destroy_buffer(buffer);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    // Cross-queue sync — transfer writes, graphics reads via wait_for
    // This is the real stress test for Counter encoding and semaphore waits
    #[test]
    fn test_cross_queue_sync() {
        init();
        let buffer = crate::create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: 4 * 64,
            memory_type: MemoryType::HostVisible,
        });

        // Transfer queue writes
        let mut transfer_cmd = crate::record(QueueType::Transfer);
        transfer_cmd.fill_buffer(&buffer, 0, 4 * 64, 0xCAFEBABE);
        let transfer_counter = crate::submit(&[transfer_cmd]);

        // Graphics queue waits for transfer before its own fill
        let mut gfx_cmd = crate::record(QueueType::Graphics);
        gfx_cmd.wait_for(transfer_counter, PipelineStage::ALL_COMMANDS);
        gfx_cmd.fill_buffer(&buffer, 0, 4 * 64, 0x12345678);
        let gfx_counter = crate::submit(&[gfx_cmd]);

        crate::wait(gfx_counter);

        let data: &[u32] = buffer.as_slice();
        assert!(data.iter().all(|&x| x == 0x12345678));

        crate::destroy_buffer(buffer);
    }

    // Hammer the pool ring — submit without waiting to force pool growth,
    // then drain and verify pools get reclaimed
    #[test]
    fn test_pool_ring_growth_and_reuse() {
        init();

        const COUNT: usize = 12; // more than initial ring size of 3
        let buffers: Vec<_> = (0..COUNT)
            .map(|_| {
                crate::create_buffer(&BufferDescription {
                    usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                    size: 4 * 16,
                    memory_type: MemoryType::HostVisible,
                })
            })
            .collect();

        // Submit all without waiting — forces ring to grow past initial 3
        let counters: Vec<_> = buffers
            .iter()
            .map(|buf| {
                let mut cmd = crate::record(QueueType::Graphics);
                cmd.fill_buffer(buf, 0, 4 * 16, 0xABCD);
                crate::submit(&[cmd])
            })
            .collect();

        // Now drain
        for c in counters {
            crate::wait(c);
        }

        // After draining, resubmit — pools should be reclaimed, not growing forever
        for buf in &buffers {
            let data: &[u32] = buf.as_slice();
            assert!(data.iter().all(|&x| x == 0xABCD));
        }

        for buf in buffers {
            crate::destroy_buffer(buf);
        }
    }
}
