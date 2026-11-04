use sgpu::{AcquiredImage, Counter, GlobalBarrier, ImageBarrier, ImageLayout, PipelineStage, QueryPool};
use smallvec::SmallVec;

use super::ResourcePool;
use super::compile::sync_2::{ExecutionBatch, SubmitId, SubmitPlan};
use super::timing::NodeTimingState;
use crate::{
    graph::{Heap, NodeDeclaration},
    node::ExecutionContext,
};

/// Counters produced by the submits recorded so far in this frame.
struct FrameState {
    counters: [Vec<Option<Counter>>; 3],
    last: [Option<Counter>; 3],
    last_frame_complete_counter: Option<Counter>,
    present_counter: Option<Counter>,
}

impl FrameState {
    fn new(plan: &SubmitPlan) -> FrameState {
        FrameState {
            counters: std::array::from_fn(|queue| vec![None; plan.submits[queue].len()]),
            last: [None, None, None],
            last_frame_complete_counter: None,
            present_counter: None,
        }
    }
}

/// Everything one frame contributes to its submits
struct FrameCtx<'a, D> {
    pool: &'a ResourcePool,
    acquired: Option<&'a AcquiredImage>,
    data: &'a D,
    curr: usize,
    frame_fresh: bool,
    previous_frame_counters: [Option<Counter>; 3],
    timing: Option<&'a QueryPool>,
}

/// How far execution has progressed across the frames in flight.
pub(crate) struct RuntimeInfo {
    pub(crate) executed_once: bool,
    pub(crate) curr_frame: usize,
    pub(crate) per_frame_counters: Vec<[Option<Counter>; 3]>,
    pub(crate) frame_executed: Vec<bool>,
}

impl RuntimeInfo {
    pub(crate) fn new(frames_in_flight: usize) -> Self {
        Self {
            executed_once: false,
            curr_frame: 0,
            per_frame_counters: vec![[None, None, None]; frames_in_flight],
            frame_executed: vec![false; frames_in_flight],
        }
    }
}

pub(crate) struct CompiledGraph<'g, R, D> {
    pub(crate) declarations: Vec<NodeDeclaration<'g, R, D>>,
    pub(crate) plan: SubmitPlan,
    pub(crate) heaps: Vec<Heap>,
    pub(crate) runtime: RuntimeInfo,
}

impl<'g, R: 'g, D: 'g> CompiledGraph<'g, R, D> {
    pub(crate) fn run(&mut self, pool: &mut ResourcePool, timing_enabled: bool, timing_state: &mut Option<NodeTimingState>, acquired: Option<&AcquiredImage>, data: &D) -> Counter {
        let slot_count = self.runtime.per_frame_counters.len();

        if let Some(acquired) = acquired {
            assert!(acquired.frame_index() < slot_count, "acquired slot {} out of range for slot count {}", acquired.frame_index(), slot_count);
            self.runtime.curr_frame = acquired.frame_index();
        } else {
            self.runtime.curr_frame = (self.runtime.curr_frame + 1) % slot_count;
        }

        let curr = self.runtime.curr_frame;
        let slot_fresh = !self.runtime.frame_executed[curr];
        let seeded = self.runtime.per_frame_counters[(curr + slot_count - 1) % slot_count];
        let ring = self.runtime.per_frame_counters[curr];

        self.ensure_timing_pools(timing_enabled, timing_state, self.declarations.len());
        pool.refresh_views();

        let timing = self.timing_pool(timing_enabled, timing_state, curr);
        let mut frame = FrameState::new(&self.plan);

        {
            let ctx = FrameCtx {
                pool,
                acquired,
                data,
                curr,
                frame_fresh: slot_fresh,
                previous_frame_counters: ring,
                timing,
            };

            self.drain(&ctx, &mut frame);
        }

        for queue in 0..3 {
            self.runtime.per_frame_counters[curr][queue] = frame.last[queue].or(seeded[queue]);
        }
        self.runtime.frame_executed[curr] = true;
        self.runtime.executed_once = true;

        self.mark_timing_written(timing_enabled, timing_state, curr);
        frame.present_counter.or(frame.last_frame_complete_counter).expect("graph has no nodes")
    }

    /// Records every submit in an order that respects the plan.
    ///
    /// Each submit's in-degree is the number of `signals` edges pointing at it,
    /// counted from the plan itself so the two can never disagree. A submit with
    /// nothing left to wait for is recorded immediately, then its counter is
    /// forwarded along its `signals`, and the next wave of ready submits is
    /// drained. The plan is acyclic, so this reaches every submit.
    fn drain(&self, ctx: &FrameCtx<'_, D>, frame: &mut FrameState) {
        let plan = &self.plan;

        let mut in_degree: [Vec<usize>; 3] = std::array::from_fn(|queue| vec![0; plan.submits[queue].len()]);
        for column in &plan.submits {
            for submit in column {
                for signal in &submit.signals {
                    in_degree[signal.queue][signal.index] += 1;
                }
            }
        }

        let mut ready: Vec<SubmitId> = Vec::new();
        for (queue, degrees) in in_degree.iter().enumerate() {
            for (index, degree) in degrees.iter().enumerate() {
                if *degree == 0 {
                    ready.push(SubmitId {
                        queue,
                        index,
                    });
                }
            }
        }

        let mut drained = 0;
        while let Some(id) = ready.pop() {
            self.record_submit(ctx, frame, id);
            drained += 1;

            for &signal in &plan.submits[id.queue][id.index].signals {
                let degree = &mut in_degree[signal.queue][signal.index];
                *degree -= 1;
                if *degree == 0 {
                    ready.push(signal);
                }
            }
        }

        let planned: usize = plan.submits.iter().map(|column| column.len()).sum();
        debug_assert_eq!(drained, planned, "plan must be acyclic: every submit drains");
    }

    fn record_submit(&self, ctx: &FrameCtx<'_, D>, frame: &mut FrameState, id: SubmitId) {
        let submit = &self.plan.submits[id.queue][id.index];

        let mut waits: SmallVec<[(Counter, PipelineStage); 3]> = SmallVec::new();
        for wait in &submit.waits {
            // The drain records sources before their waiters, so the counter is
            // always in place by the time it is needed.
            let counter = frame.counters[wait.source.queue][wait.source.index].expect("a submit only drains after its sources");
            waits.push((counter, wait.stage));
        }
        for slot in &submit.previous_waits {
            if let Some(counter) = ctx.previous_frame_counters[slot.queue as usize] {
                waits.push((counter, slot.stage));
            }
        }

        let mut cmds = Vec::with_capacity(submit.exec_batches.len());
        for (position, batch) in submit.exec_batches.iter().enumerate() {
            let mut cmd = sgpu::record(submit.queue);

            // Every command buffer folds into a single `vkQueueSubmit`, so the
            // waits only need recording once — on the first batch.
            if position == 0 {
                if submit.acquire
                    && let Some(acquired) = ctx.acquired
                {
                    cmd.wait_for_swapchain_image(acquired);
                }
                for &(counter, stage) in &waits {
                    cmd.wait_for(counter, stage);
                }
            }

            self.record_barriers(ctx, &mut cmd, batch);

            {
                let mut exec = ExecutionContext {
                    cmd: &mut cmd,
                    pool: ctx.pool,
                    curr_frame: ctx.curr,
                };
                for &node in &batch.nodes {
                    let first = (node as u32) * 2;

                    if let Some(pool) = ctx.timing {
                        exec.cmd.reset_query_pool(pool, first, 2);
                        exec.cmd.write_timestamp(pool, first, PipelineStage::ALL_COMMANDS);
                    }

                    self.declarations[node].built.execute(ctx.data, &mut exec);

                    if let Some(pool) = ctx.timing {
                        exec.cmd.write_timestamp(pool, first + 1, PipelineStage::ALL_COMMANDS);
                    }
                }
            }

            cmds.push(cmd);
        }

        let counter = sgpu::submit(&cmds);
        frame.counters[id.queue][id.index] = Some(counter);
        frame.last[id.queue] = Some(counter);
        frame.last_frame_complete_counter = Some(counter);
        if submit.terminal {
            frame.present_counter = Some(counter);
        }
    }

    fn record_barriers(&self, ctx: &FrameCtx<'_, D>, cmd: &mut sgpu::CommandBuffer, batch: &ExecutionBatch) {
        // A slot that has never run has no previous state to order against, so
        // its barriers transition from nothing rather than from whatever the
        // last group on some other slot left behind.
        let planned = if ctx.frame_fresh {
            &batch.first_frame_barriers
        } else {
            &batch.before_barriers
        };

        let images: SmallVec<[ImageBarrier; 8]> = planned
            .images
            .iter()
            .map(|image| ImageBarrier {
                view: ctx.pool.get_image(image.image).image_at(ctx.curr).default_view(),
                previous_accesses: &image.previous,
                next_accesses: &image.next,
                previous_layout: ImageLayout::Optimal,
                next_layout: ImageLayout::Optimal,
                discard_contents: image.discard,
            })
            .collect();

        if images.is_empty() && planned.global.previous.is_empty() {
            return;
        }

        let global = (!planned.global.previous.is_empty()).then_some(GlobalBarrier {
            previous_accesses: &planned.global.previous,
            next_accesses: &planned.global.next,
        });
        cmd.barriers(global.as_ref(), &images);
    }

    /// Create the timestamp pools the current timing settings need.
    fn ensure_timing_pools(&self, enabled: bool, timing: &mut Option<NodeTimingState>, node_count: usize) {
        if !enabled || node_count == 0 {
            return;
        }

        let slot_count = self.runtime.per_frame_counters.len();
        let query_count = u32::try_from(node_count.checked_mul(2).expect("node count overflows usize")).expect("too many nodes for timestamp queries");

        if timing.as_ref().is_some_and(|timing| timing.query_count == query_count && timing.pools.len() == slot_count) {
            return;
        }

        if timing.is_some() {
            sgpu::wait_idle();
        }

        *timing = Some(NodeTimingState::new(slot_count, query_count));
    }

    /// The timestamp pool of `slot`, or `None` when timing is off.
    fn timing_pool<'a>(&self, enabled: bool, timing: &'a Option<NodeTimingState>, slot: usize) -> Option<&'a QueryPool> {
        if !enabled {
            return None;
        }

        timing.as_ref().and_then(|timing| timing.pools.get(slot))
    }

    fn mark_timing_written(&self, enabled: bool, timing: &mut Option<NodeTimingState>, slot: usize) {
        if enabled
            && let Some(timing) = timing
            && let Some(written) = timing.written.get_mut(slot)
        {
            *written = true;
        }
    }
}
