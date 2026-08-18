use serde::{Serialize, de::DeserializeOwned};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub type CacheResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn ensure_parent_dir(path: impl AsRef<Path>) -> std::io::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub fn atomic_write(path: impl AsRef<Path>, bytes: &[u8]) -> std::io::Result<()> {
    let path = path.as_ref();
    ensure_parent_dir(path)?;

    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown_host".to_string());
    let job = std::env::var("SLURM_JOB_ID").unwrap_or_else(|_| "nojob".to_string());
    let array_task = std::env::var("SLURM_ARRAY_TASK_ID").unwrap_or_else(|_| "notask".to_string());

    let mut tmp = PathBuf::from(path);
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("cache");

    tmp.set_file_name(format!(
        "{}.tmp.{}.{}.{}.{}",
        file_name,
        host,
        job,
        array_task,
        std::process::id()
    ));

    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }

    fs::rename(tmp, path)?;
    Ok(())
}

pub fn save_bincode_atomic<T: Serialize>(path: impl AsRef<Path>, value: &T) -> CacheResult<()> {
    let encoded = bincode::serialize(value)?;
    atomic_write(path, &encoded)?;
    Ok(())
}

pub fn load_bincode<T: DeserializeOwned>(path: impl AsRef<Path>) -> CacheResult<T> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}
