import random
import torch


class LabelStream:
    '''Base class for iterators that determine from which context should be sampled.'''

    def __init__(self):
        self.n_contexts = None

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError  # returns 'context_id' (i.e., starting from context 0); type=int


class SharpBoundaryStream(LabelStream):
    '''Set up a label-stream with strictly separated contexts (as in the academic continual learning setting).'''

    def __init__(self, n_contexts, iters_per_context):
        '''Instantiate the dissociated-stream object by defining its parameters.
        Args:
            n_contexts (int): number of contexts
            iters_per_context (int): number of iterations to generate per context
        '''

        super().__init__()
        self.n_contexts = n_contexts
        self.iters_per_context = iters_per_context

        # For keeping track of context
        self.iters_count = 0
        self.context = 0

    def __next__(self):
        self.iters_count += 1
        # -move to next context when all iterations of current context are done
        if self.iters_count > self.iters_per_context:
            self.iters_count = 1
            self.context += 1
            if self.context >= self.n_contexts:
                raise StopIteration
        next_label = self.context
        return next_label


class RandomStream(LabelStream):
    '''Set up a completely random label-stream.'''

    def __init__(self, n_contexts):
        super().__init__()
        self.n_contexts = n_contexts

    def __next__(self):
        return random.randint(0, self.n_contexts-1)


def _linear_line(number, direction="up"):
    if direction == "up":
        return torch.FloatTensor([(i+0.5) / number for i in range(number)])
    return torch.FloatTensor([1 - ((i+0.5) / number) for i in range(number)])

def _probs_per_context(n_contexts, iters_per_context, context_id, fuzziness=3):
    if (2 * fuzziness) > iters_per_context:
        raise ValueError("Fuzziness must be smaller than half the number of iterations per context.")

    # Start with zero probability for every iteration
    probs = torch.zeros(n_contexts * iters_per_context)

    # Depending on which context, add non-zero probabilities
    if context_id == 0:
        # -first period of seeing context 0
        end = int(iters_per_context / 2)
        probs[0:(end - fuzziness)].add_(1)
        probs[(end - fuzziness):(end + fuzziness)] = _linear_line(2 * fuzziness, direction="down")
        # -second period of seeing context 0
        start = int(iters_per_context / 2) + (n_contexts - 1) * iters_per_context
        probs[(start - fuzziness):(start + fuzziness)] = _linear_line(2 * fuzziness, direction="up")
        probs[(start + fuzziness):(iters_per_context * n_contexts)].add_(1)
    else:
        start = int(iters_per_context / 2) + (context_id - 1) * iters_per_context
        end = int(iters_per_context / 2) + context_id * iters_per_context
        probs[(start - fuzziness):(start + fuzziness)] = _linear_line(2 * fuzziness, direction="up")
        probs[(start + fuzziness):(end - fuzziness)].add_(1)
        probs[(end - fuzziness):(end + fuzziness)] = _linear_line(2 * fuzziness, direction="down")

    return probs

class FuzzyBoundaryStream(LabelStream):
    '''Set up a label-stream for an experiment with fuzzy context boundaries.'''

    def __init__(self, n_contexts, iters_per_context, fuzziness, batch_size=1):
        super().__init__()
        self.n_contexts = n_contexts
        self.batch_size = batch_size
        self.total_iters = iters_per_context*n_contexts
        self.batch_count = 0
        self.iters_count = 0

        # For each context, get a tensor with its probability per iteration
        context_probs_per_iter = [_probs_per_context(
            n_contexts, iters_per_context, context_id, fuzziness=fuzziness
        ) for context_id in range(n_contexts)]

        # For each iteration, specify a probability-distribution over the contexts
        self.context_probs = []
        context_probs_tensor = torch.cat(context_probs_per_iter).view(n_contexts, iters_per_context*n_contexts)
        for iter_id in range(iters_per_context*n_contexts):
            self.context_probs.append(context_probs_tensor[:, iter_id])

    def __next__(self):
        self.batch_count += 1
        # -move to next iteration when all mini-batch samples of current iteration are done
        if self.batch_count > self.batch_size:
            self.batch_count = 1
            self.iters_count += 1
            if self.iters_count >= self.total_iters:
                raise StopIteration
        # -sample a context label using the probability-distribution of current iteration
        context_label = random.choices(range(self.n_contexts), self.context_probs[self.iters_count])[0]
        return context_label
