def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    _______________________________________________________
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    _______________________________________________________
    """

    s = [word.strip(" '\"\n").lower() for word in words.strip("[]").split(",")]

    from gensim import downloader as api
    from gensim.models import KeyedVectors

    import random

    # Load the smaller word vector model from gensim
    word_vectors = KeyedVectors.load("/home/ugrads/n/nrs2586/datathon/starter_code/word_vectors.model")
    current_round = len(previousGuesses)
    guess = [s[i] for i in range(4)]
    endTurn = False
    val = 0
    ctr = 0
    l = ["eye"]
    for word1 in l:
        ctr += 1
        if ctr > 40000:
            break
        a = []
        for word2 in s:
            if word1 in word_vectors and word2 in word_vectors:
                try:
                    similarity = word_vectors.similarity(word1, word2)
                    a.append((similarity, word2))
                except KeyError:
                    print(f"One of the words '{word1}' or '{word2}' is not in the vocabulary.")
                    endTurn = True
                    break
            else:
                print(f"One of the words '{word1}' or '{word2}' is missing in the vocabulary.")
                endTurn = True
                break
        if endTurn:
            break
        a.sort(reverse=True, key=lambda x: x[0])
        print(a)
        if a[3][0] > val:
            val = a[3][0]
            guess = [a[0][1], a[1][1], a[2][1], a[3][1]]
            print(val)
        
    return guess, endTurn
