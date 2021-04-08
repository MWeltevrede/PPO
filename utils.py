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


def update(existingAggregate, newValue):
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