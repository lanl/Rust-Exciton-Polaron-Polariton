#[macro_export]
macro_rules! timed_println {
    ($($arg:tt)*) => {{
        let now = chrono::Local::now();
        println!("[{}] {}", now.format("%Y-%m-%d %H:%M:%S%.3f"), format!($($arg)*));
    }};
}

pub fn init_logging() {
    // Any initialization code can go here
    timed_println!("Logging initialized");
}