import scipy.signal
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def update_single(existingAggregate, newValue):
    """
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    
    Updates an aggregete used for calculation of the mean and sample variance.
    input: 
        aggregate, new value
    output:
        aggregate
    """
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

def update_sets(aggregate_A, aggregate_B):
    """
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    
    Combines the aggregates of two sets of data
    input: 
        aggregate A, aggregate B
    output:
        aggregate AB
    """
    (n_A, m_A, M2_A) = aggregate_A
    (n_B, m_B, M2_B) = aggregate_B
    
    n = n_A + n_B
    mean = (n_A*m_A + n_B*m_B) / n
    
    delta = m_B - m_A
    M2 = M2_A + M2_B + delta ** 2 * n_A * n_B / n
    
    return (n, mean, M2)


def finalize(existingAggregate):
    """
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    
    Calculates the sample mean and sample variance from an aggregate.
    input: 
        aggregate
    output:
        (sample mean, sample variance)
    """
    (count, mean, M2) = existingAggregate
    if count < 2:
        return (0,1)
    else:
        (mean, sampleVariance) = (mean, M2 / (count - 1))
        return (mean, sampleVariance)