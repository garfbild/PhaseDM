use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

#[derive(Clone)]
pub struct CumulativeTimer {
    timers: Arc<Mutex<HashMap<String, Duration>>>,
    start_times: Arc<Mutex<HashMap<String, Instant>>>,
}

impl CumulativeTimer {
    pub fn new() -> Self {
        CumulativeTimer {
            timers: Arc::new(Mutex::new(HashMap::new())),
            start_times: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn start(&self, section_name: &str) {
        let mut start_times = self.start_times.lock().unwrap();
        start_times.insert(section_name.to_string(), Instant::now());
    }

    pub fn stop(&self, section_name: &str) {
        let now = Instant::now();
        let section_name = section_name.to_string();
        
        // Get the start time and remove it
        let start_time = {
            let mut start_times = self.start_times.lock().unwrap();
            start_times.remove(&section_name).expect("Timer was never started")
        };
        
        // Calculate duration and add to cumulative time
        let duration = now.duration_since(start_time);
        let mut timers = self.timers.lock().unwrap();
        let entry = timers.entry(section_name).or_insert(Duration::new(0, 0));
        *entry += duration;
    }

    pub fn report(&self) -> String {
        let timers = self.timers.lock().unwrap();
        let mut report = String::from("Cumulative timing report:\n");
        
        // Convert timers to a sortable vector
        let mut times: Vec<(&String, &Duration)> = timers.iter().collect();
        
        // Sort by duration (descending)
        times.sort_by(|a, b| b.1.cmp(a.1));
        
        // Build the report
        for (name, duration) in times {
            let secs = duration.as_secs_f64();
            report.push_str(&format!("{}: {:.6} seconds\n", name, secs));
        }
        
        report
    }
    
    pub fn reset(&self) {
        let mut timers = self.timers.lock().unwrap();
        timers.clear();
    }
}

// Create a static global timer
lazy_static::lazy_static! {
    pub static ref GLOBAL_TIMER: CumulativeTimer = CumulativeTimer::new();
}

// Macro to time a section
#[macro_export]
macro_rules! time_section {
    ($section:expr, $code:expr) => {{
        $crate::timing::GLOBAL_TIMER.start($section);
        let result = $code;
        $crate::timing::GLOBAL_TIMER.stop($section);
        result
    }};
}

// Function to get the timing report
pub fn get_timing_report() -> String {
    GLOBAL_TIMER.report()
}

// Function to reset all timers
pub fn reset_timers() {
    GLOBAL_TIMER.reset();
}