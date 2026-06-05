import os

def load_dictionary(
    first_letter
):

    dictionary_path = os.path.join(
        "dictionary",
        f"{first_letter.lower()}.txt"
    )

    words = []

    try:

        with open(
            dictionary_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                word = line.strip()

                if word:
                    words.append(word)

    except FileNotFoundError:

        return []

    return words

def levenshtein_distance(
    s1,
    s2
):

    len1 = len(s1)
    len2 = len(s2)

    matrix = [
        [0] * (len2 + 1)
        for _ in range(len1 + 1)
    ]

    for i in range(len1 + 1):
        matrix[i][0] = i

    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):

        for j in range(1, len2 + 1):

            cost = (
                0
                if s1[i - 1] == s2[j - 1]
                else 1
            )

            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )

    return matrix[len1][len2]

def get_suggestions(
    word,
    character_alternatives,
    max_suggestions=20
):

    if not word:
        return []

    dictionary_words = load_dictionary(
        word[0]
    )

    suggestions = []

    for dictionary_word in dictionary_words:
        
        if len(dictionary_word) != len(word):
            continue

        distance = levenshtein_distance(
            word.lower(),
            dictionary_word.lower()
        )

        ocr_score = 0

        for i in range(
            min(
                len(dictionary_word),
                len(character_alternatives)
            )
        ):

            candidate_char = dictionary_word[i].lower()

            matched = False

            for char, confidence in character_alternatives[i]:

                if candidate_char == str(char).lower():

                    ocr_score -= confidence
                    matched = True
                    break

            if not matched:
                ocr_score += 100

        score = (
            distance,
            ocr_score,
        )

        suggestions.append(
            (
                dictionary_word,
                score
            )
        )

    suggestions.sort(
        key=lambda x: x[1]
    )

    return suggestions[:max_suggestions]

def get_best_match(
    word,
    character_alternatives
):

    suggestions = get_suggestions(
        word,
        character_alternatives
    )

    if len(suggestions) == 0:
        return word

    return suggestions[0][0]