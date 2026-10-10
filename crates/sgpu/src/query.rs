use ash::vk;

pub struct QueryPool {
    pub(crate) raw: vk::QueryPool,
    pub(crate) query_count: u32,
}

impl QueryPool {
    pub fn query_count(&self) -> u32 { return self.query_count; }

    pub fn timestamp_period(&self) -> f32 { return crate::CONTEXT.get().expect("Not initialized").device.timestamp_period; }

    pub fn get_timestamps(&self, first_query: u32, query_count: u32) -> Vec<u64> {
        assert!((first_query + query_count) <= self.query_count, "Query range out of bounds");

        let mut data = vec![0u64; query_count as usize];
        crate::CONTEXT.get().expect("Not initialized").device.get_query_results_u64(self.raw, first_query, &mut data);

        return data;
    }

    pub fn get_timestamps_ns(&self, first_query: u32, query_count: u32) -> Vec<f64> {
        let period = self.timestamp_period() as f64;
        return self.get_timestamps(first_query, query_count).into_iter().map(|ticks| ticks as f64 * period).collect();
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) { crate::CONTEXT.get().unwrap().device.destroy_query_pool(self.raw); }
}
