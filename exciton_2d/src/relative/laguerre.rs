pub fn factorial(n: i32) -> u64 {
    if n <= 1 {
        return 1;
    }

    let mut result = 1;

    for i in 2..=n {
        result *= i as u64;
    }

    result
}

pub fn binomial(n: i32, k: i32) -> u64 {
    if k < 0 || k > n {
        return 0;
    }

    if k == 0 || k == n {
        return 1;
    }

    let k = std::cmp::min(k, n - k);
    let mut result = 1;

    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }

    result
}

pub fn laguerre_assoc(n: i32, alpha: i32, x: f64) -> f64 {
    let mut result = 0.0;

    for i in 0..=n {
        let binomial = binomial(n + alpha, n - i);
        result += binomial as f64 * (-x).powi(i) / factorial(i) as f64;
    }

    result
}
