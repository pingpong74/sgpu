use sgpu::{AccessType, AcquiredImage, Counter, GlobalBarrier, ImageBarrier, ImageLayout, PipelineStage, QueryPool};
use smallvec::SmallVec;

use super::TaskGraph;
use crate::node::ExecutionContext;

use super::ErasedNode;
use crate::graph::compile::sync::{CompiledImageBarrier, SubmissionPlan};

struct FrameState {
    counters: [Vec<Option<Counter>>; 3],
    head: [usize; 3],
    last: [Option<Counter>; 3],
    // last submission, (completion counter of the frame)
    last_counter: Option<Counter>,
    present_counter: Option<Counter>,
}

impl FrameState {
    fn new(plan: &SubmissionPlan) -> FrameState {
        FrameState {
            counters: std::array::from_fn(|queue| vec![None; plan.submissions[queue].len()]),
            head: [0; 3],
            last: [None; 3],
            last_counter: None,
            present_counter: None,
        }
    }
}

// Everything one submission record needs
struct Checkpoints<'a, 'g, R: 'g, D: 'g> {
    graph: &'a TaskGraph<'g, R, D>,
    built: &'a [Box<dyn ErasedNode<R, D> + 'g>],
    plan: &'a SubmissionPlan,
    acquired: Option<&'a AcquiredImage>,
    data: &'a D,
    curr: usize,
    graph_has_run: bool,
    slot_has_run: bool,
    // the previous frame's last counters per queue.
    ring: [Option<Counter>; 3],
    timing: Option<&'a QueryPool>,
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    pub(super) fn run(&mut self, acquired: Option<&AcquiredImage>, data: &D) -> Counter {
        let mut compiled = self.compiled.take().expect("TaskGraph is not compiled");

        if let Some(acquired) = acquired {
            assert!(acquired.frame_index() < self.frames_in_flight, "acquired slot {} out of range for slot_count {}", acquired.frame_index(), self.frames_in_flight);
            compiled.runtime.curr_frame = acquired.frame_index();
        } else if self.frames_in_flight == 0 {
            panic!("TaskGraph::execute requires set_frames_in_flight(F)");
        } else {
            compiled.runtime.curr_frame = (compiled.runtime.curr_frame + 1) % self.frames_in_flight;
        }

        let curr = compiled.runtime.curr_frame;
        let graph_has_run = compiled.runtime.executed_once;
        let slot_fresh = !compiled.runtime.frame_executed[curr];
        let seeded = compiled.runtime.per_frame_counters[(curr + self.frames_in_flight - 1) % self.frames_in_flight];
        let ring = compiled.runtime.per_frame_counters[curr];

        self.ensure_timing_pools(compiled.built.len());
        self.views.refresh(&self.images);

        let timing = self.timing_pool(curr);
        let mut frame = FrameState::new(&compiled.submissions);

        {
            let checkpoints = Checkpoints {
                graph: self,
                built: &compiled.built,
                plan: &compiled.submissions,
                acquired,
                data,
                curr,
                graph_has_run,
                slot_has_run: slot_fresh,
                ring,
                timing,
            };

            for &(queue, index) in &compiled.submissions.order {
                flush_until(&checkpoints, &mut frame, queue as usize, index);
            }
            for (queue, head) in frame.head.iter().enumerate() {
                debug_assert_eq!(*head, checkpoints.plan.submissions[queue].len(), "record order skipped a queue tail");
            }
        }

        for (queue, _) in compiled.submissions.submissions.iter().enumerate() {
            compiled.runtime.per_frame_counters[curr][queue] = frame.last[queue].or(seeded[queue]);
        }
        compiled.runtime.frame_executed[curr] = true;
        compiled.runtime.executed_once = true;

        self.compiled = Some(compiled);
        self.mark_timing_written(curr);
        frame.present_counter.or(frame.last_counter).expect("graph has no nodes")
    }
}

fn flush_until<'g, R: 'g, D: 'g>(checkpoints: &Checkpoints<'_, 'g, R, D>, frame: &mut FrameState, queue: usize, target: usize) {
    debug_assert!(queue < 3, "queue index {queue} out of the registry");
    debug_assert!(target < checkpoints.plan.submissions[queue].len(), "submission {target} referenced out of queue {queue}'s plan");
    while frame.head[queue] <= target {
        record_submission(checkpoints, frame, queue, frame.head[queue]);
        frame.head[queue] += 1;
    }
}

fn record_submission<'g, R: 'g, D: 'g>(checkpoints: &Checkpoints<'_, 'g, R, D>, frame: &mut FrameState, queue: usize, index: usize) {
    let submission = &checkpoints.plan.submissions[queue][index];

    let mut waits: SmallVec<[(Counter, PipelineStage); 3]> = SmallVec::new();

    for wait in &submission.waits {
        let source_queue = wait.queue as usize;
        flush_until(checkpoints, frame, source_queue, wait.source);
        let counter = frame.counters[source_queue][wait.source].expect("exact wait source not submitted");
        waits.push((counter, wait.stage));
    }

    for wait in &submission.slot_waits {
        if let Some(counter) = checkpoints.ring[wait.queue as usize] {
            waits.push((counter, wait.stage));
        }
    }

    let mut cmd = sgpu::record(submission.queue);

    if submission.acquire
        && let Some(acquired) = checkpoints.acquired
    {
        cmd.wait_for_swapchain_image(acquired);
    }
    for (counter, stage) in waits {
        cmd.wait_for(counter, stage);
    }

    record_barriers(checkpoints, &mut cmd, submission);

    {
        let mut ctx = ExecutionContext {
            cmd: &mut cmd,
            images: &checkpoints.graph.images,
            buffers: &checkpoints.graph.buffers,
            views: &checkpoints.graph.views,
            curr_frame: checkpoints.curr,
        };
        for &node in &submission.nodes {
            let first = (node as u32) * 2;

            if let Some(pool) = checkpoints.timing {
                ctx.cmd.reset_query_pool(pool, first, 2);
                ctx.cmd.write_timestamp(pool, first, PipelineStage::ALL_COMMANDS);
            }

            checkpoints.built[node].execute(checkpoints.data, &mut ctx);

            if let Some(pool) = checkpoints.timing {
                ctx.cmd.write_timestamp(pool, first + 1, PipelineStage::ALL_COMMANDS);
            }
        }
    }

    let counter = sgpu::submit(&[cmd]);
    frame.counters[queue][index] = Some(counter);
    frame.last[queue] = Some(counter);
    frame.last_counter = Some(counter);
    if submission.terminal {
        frame.present_counter = Some(counter);
    }
}

fn record_barriers<'g, R: 'g, D: 'g>(checkpoints: &Checkpoints<'_, 'g, R, D>, cmd: &mut sgpu::CommandBuffer, submission: &super::compile::sync::Submission) {
    let barriers = &submission.barriers;
    let mut images: SmallVec<[ImageBarrier; 8]> = SmallVec::new();

    if checkpoints.graph_has_run {
        for planned in &barriers.steady.images {
            let barrier = if checkpoints.slot_has_run {
                planned.first_use.as_ref().unwrap_or(&planned.barrier)
            } else {
                &planned.barrier
            };
            images.push(image_barrier(checkpoints, barrier));
        }
        let global = &barriers.steady.global;
        record_barrier_list(cmd, &global.previous, &global.next, &images);
    } else {
        for barrier in &barriers.first_frame.images {
            images.push(image_barrier(checkpoints, barrier));
        }
        let global = &barriers.first_frame.global;
        record_barrier_list(cmd, &global.previous, &global.next, &images);
    }
}

fn image_barrier<'a, 'g, R: 'g, D: 'g>(checkpoints: &Checkpoints<'_, 'g, R, D>, barrier: &'a CompiledImageBarrier) -> ImageBarrier<'a> {
    ImageBarrier {
        view: checkpoints.graph.images[barrier.image as usize].image_at(checkpoints.curr).default_view(),
        previous_accesses: &barrier.previous,
        next_accesses: &barrier.next,
        previous_layout: ImageLayout::Optimal,
        next_layout: ImageLayout::Optimal,
        discard_contents: barrier.discard,
    }
}

fn record_barrier_list(cmd: &mut sgpu::CommandBuffer, previous: &[AccessType], next: &[AccessType], images: &[ImageBarrier]) {
    if images.is_empty() && previous.is_empty() {
        return;
    }

    let global = (!previous.is_empty()).then_some(GlobalBarrier {
        previous_accesses: previous,
        next_accesses: next,
    });
    cmd.barriers(global.as_ref(), images);
}
