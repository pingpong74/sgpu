use sgpu::{AcquiredImage, CommandBuffer, Counter, GlobalBarrier, ImageBarrier, ImageLayout, PipelineStage, QueueType};
use smallvec::SmallVec;

use super::TaskGraph;
use crate::node::ExecutionContext;

struct Frame {
    open: [Option<CommandBuffer>; 3],
    counters: [Option<Counter>; 3],
    last: Option<Counter>,
}

impl Frame {
    fn flush(&mut self, q: usize) {
        if let Some(cmd) = self.open[q].take() {
            let counter = sgpu::submit(&[cmd]);
            self.counters[q] = Some(counter);
            self.last = Some(counter);
        }
    }
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    pub(super) fn run(&mut self, acquired: Option<&AcquiredImage>, data: &D) -> Counter {
        let steady = self.executed_once;
        let mut frame = Frame {
            open: [None, None, None],
            counters: self.prev_counters,
            last: None,
        };
        let mut present_counter = None;

        let compiled = self.compiled.as_ref().expect("graph compiled before run");
        let present_batch_index = compiled.present.then(|| compiled.batches.len() - 1);

        for (batch_index, queue_batches) in compiled.batches.iter().enumerate() {
            let present_batch = present_batch_index == Some(batch_index);

            for queue_batch in queue_batches {
                let q = queue_batch.queue as usize;

                let mut waits: SmallVec<[(Counter, PipelineStage); 3]> = SmallVec::new();
                let wrap: &[(QueueType, PipelineStage)] = if steady {
                    &queue_batch.prev_frame_waits
                } else {
                    &[]
                };
                for &(src, stage) in queue_batch.waits.iter().chain(wrap) {
                    let src = src as usize;
                    frame.flush(src);
                    waits.push((frame.counters[src].expect("cross-queue dependency on a queue that never submitted"), stage));
                }

                if present_batch || queue_batch.acquire || !waits.is_empty() {
                    frame.flush(q);
                }
                let cmd = frame.open[q].get_or_insert_with(|| sgpu::record(queue_batch.queue));

                if queue_batch.acquire
                    && let Some(acquired) = acquired
                {
                    cmd.wait_for_swapchain_image(acquired);
                }
                for (counter, stage) in waits {
                    cmd.wait_for(counter, stage);
                }

                let barriers = if steady {
                    &queue_batch.barriers
                } else {
                    &queue_batch.undefined_barriers
                };
                if !barriers.images.is_empty() || !barriers.global.previous.is_empty() {
                    let global = (!barriers.global.previous.is_empty()).then(|| GlobalBarrier {
                        previous_accesses: &barriers.global.previous,
                        next_accesses: &barriers.global.next,
                    });

                    let mut imgs: SmallVec<[ImageBarrier; 8]> = SmallVec::new();
                    for barrier in &barriers.images {
                        imgs.push(ImageBarrier {
                            view: self.images[barrier.image as usize].image().default_view(),
                            previous_accesses: &barrier.previous,
                            next_accesses: &barrier.next,
                            previous_layout: ImageLayout::Optimal,
                            next_layout: ImageLayout::Optimal,
                            discard_contents: barrier.discard,
                        });
                    }
                    cmd.barriers(global.as_ref(), &imgs);
                }

                let mut ctx = ExecutionContext {
                    cmd,
                    images: &self.images,
                    buffers: &self.buffers,
                };
                for &n in &queue_batch.nodes {
                    compiled.built[n].execute(data, &mut ctx);
                }

                if present_batch {
                    frame.flush(q);
                    present_counter = Some(frame.counters[q].expect("present submit"));
                }
            }
        }

        for q in 0..3 {
            frame.flush(q);
        }

        self.prev_counters = frame.counters;
        self.executed_once = true;

        present_counter.or(frame.last).expect("graph has no nodes")
    }
}
