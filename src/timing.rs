use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

// Global verbosity flag
lazy_static::lazy_static! {
    static ref TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
}

pub struct ThreadLocalTimer {
    timers: RefCell<HashMap<String, Duration>>,
    start_times: RefCell<HashMap<String, Instant>>,
}

// A global collector to gather results from all threads
lazy_static::lazy_static! {
    static ref GLOBAL_COLLECTOR: Arc<Mutex<HashMap<String, Duration>>> = 
        Arc::new(Mutex::new(HashMap::new()));
}

thread_local! {
    static LOCAL_TIMER: ThreadLocalTimer = ThreadLocalTimer::new();
}

impl ThreadLocalTimer {
    fn new() -> Self {
        ThreadLocalTimer {
            timers: RefCell::new(HashMap::new()),
            start_times: RefCell::new(HashMap::new()),
        }
    }

    pub fn start(section_name: &str) {
        // Skip if timing is disabled
        if !TIMING_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        
        LOCAL_TIMER.with(|timer| {
            timer.start_times.borrow_mut().insert(
                section_name.to_string(), 
                Instant::now()
            );
        });
    }

    pub fn stop(section_name: &str) {
        // Skip if timing is disabled
        if !TIMING_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        
        LOCAL_TIMER.with(|timer| {
            let now = Instant::now();
            let section_name_string = section_name.to_string();
            
            // Try to get the start time, but don't panic if it's not there
            if let Some(start_time) = timer.start_times.borrow_mut().remove(&section_name_string) {
                let duration = now.duration_since(start_time);
                
                let mut timers = timer.timers.borrow_mut();
                let entry = timers.entry(section_name_string).or_insert(Duration::new(0, 0));
                *entry += duration;
            }
        });
    }
    
    // Collect current thread's timing data into the global collector
    pub fn flush_current_thread() {
        // Skip if timing is disabled
        if !TIMING_ENABLED.load(Ordering::Relaxed) {
            return;
        }

        
        LOCAL_TIMER.with(|timer| {
            let local_timers = timer.timers.borrow();
            
            let mut global = GLOBAL_COLLECTOR.lock().unwrap();
            
            // Take the maximum duration for each section, not the sum
            for (section, duration) in local_timers.iter() {      
                global.entry(section.clone())
                    .and_modify(|existing| {
                        // Keep the maximum duration
                        if *duration > *existing {
                            *existing = *duration;
                        }
                    })
                    .or_insert(*duration);
            }
        });
    }
    
    // Collect all thread-local timers into a global report
    pub fn collect_all_reports() -> HashMap<String, Duration> {
        // Skip if timing is disabled - return empty map
        if !TIMING_ENABLED.load(Ordering::Relaxed) {
            return HashMap::new();
        }
        
        // Flush the current thread first (the one calling this method)
        ThreadLocalTimer::flush_current_thread();
        
        // Force execution on all threads to flush their data
        // Note: this may not work if the thread pool is idle
        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                s.spawn(move |_| {
                    ThreadLocalTimer::flush_current_thread();
                });
            }
        });
        
        // Return a clone of the consolidated results
        let global_data = GLOBAL_COLLECTOR.lock().unwrap();
        
        global_data.clone()
    }
    
    // Generate a formatted report string
    pub fn get_timing_report() -> String {
        // Skip if timing is disabled
        if !TIMING_ENABLED.load(Ordering::Relaxed) {
            return String::from("Timing disabled");
        }
        
        let timers = ThreadLocalTimer::collect_all_reports();
        let mut report = String::from("Cumulative timing report:\n");
        
        if timers.is_empty() {
            report.push_str("No timing data collected.");
            return report;
        }
        
        // Convert timers to a sortable vector
        let mut times: Vec<(&String, &Duration)> = timers.iter().collect();
        
        // Sort by duration (descending)
        times.sort_by(|a, b| b.1.cmp(a.1));
        
        // Build the report with raw duration values
        for (name, duration) in times {
            // Show raw duration in various units
            let nanos = duration.as_nanos();
            let micros = duration.as_micros();
            let millis = duration.as_millis();
            
            report.push_str(&format!("{}: {:?} ({} ns, {} µs, {} ms)\n", 
                                    name, duration, nanos, micros, millis));
        }
        
        report
    }
}

// Public API functions to control verbosity
pub fn enable_timing(enable: bool) {
    TIMING_ENABLED.store(enable, Ordering::Relaxed);
}

pub fn is_timing_enabled() -> bool {
    TIMING_ENABLED.load(Ordering::Relaxed)
}

// Public API functions
pub fn start_timer(section_name: &str) {
    ThreadLocalTimer::start(section_name);
}

pub fn stop_timer(section_name: &str) {
    ThreadLocalTimer::stop(section_name);
}

pub fn get_timing_report() -> String {
    ThreadLocalTimer::get_timing_report()
}

pub fn reset_timers() {
    if TIMING_ENABLED.load(Ordering::Relaxed) {
        println!("Resetting all timers");
    }
    
    // Clear global collector
    let mut global = GLOBAL_COLLECTOR.lock().unwrap();
    global.clear();
}

// Macro to time a section, with no-op when timing is disabled
#[macro_export]
macro_rules! time_section {
    ($section:expr, $code:expr) => {{
        if $crate::timing::is_timing_enabled() {
            $crate::timing::start_timer($section);
            let result = $code;
            $crate::timing::stop_timer($section);
            result
        } else {
            // Just execute the code without timing when disabled
            $code
        }
    }};
}