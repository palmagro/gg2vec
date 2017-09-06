from gensim.models import word2vec

def train_sentence_sg(model, sentence, alpha, work=None):
    """
    Update skip-gram model by training on a single sentence.

    The sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                   model.vocab[w].sample_int > model.random.rand() * 2**32]

    
#    for pos, word in enumerate(word_vocabs):
    if len(word_vocabs) > 0:
        pos = 0
        word = word_vocabs[0]
        reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
            # don't train on the `word` itself
            if pos2 != pos:
                word2vec.train_sg_pair(model, model.index2word[word.index], word2.index, alpha)

    return len(word_vocabs)

def train_sentence_cbow(model, sentence, alpha, work=None, neu1=None):
    """
    Update CBOW model by training on a single sentence.

    The sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                   model.vocab[w].sample_int > model.random.rand() * 2**32]
    pos = 0
    word = word_vocabs[0]
    context = word_vocabs[1:]
    reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
    start = max(0, pos - model.window + reduced_window)
    window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
    word2_indices = [word2.index for pos2, word2 in enumerate(context) if (word2 is not None and pos2 != pos)]
    l1 = word2vec.np_sum(model.syn0[word2_indices], axis=0)  # 1 x vector_size
    if word2_indices and model.cbow_mean:
        l1 /= len(word2_indices)
    word2vec.train_cbow_pair(model, word, word2_indices, l1, alpha)

    return len(word_vocabs)

word2vec.train_sentence_sg = train_sentence_sg
word2vec.train_sentence_cbow = train_sentence_cbow
