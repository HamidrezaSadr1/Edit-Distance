
class Edit_Distance:
    """
    Instantiate a multiplication operation.
    Numbers will be multiplied by the given multiplier.
    
    :param multiplier: The multiplier.
    :type multiplier: int
    """
    
    def __init__(self, edit_distance):
        self.edit_distance = edit_distance
    

    # hamming_ditance.py
    def hamming_distance(str1, str2):
        """
        :Definition:
         
        Hamming distance is a metric used to measure the difference between two strings of equal length. It 
        counts the number of positions at which the corresponding symbols are different. In simpler terms, it 
        calculates the number of substitutions required to make the strings identical, assuming they're of the 
        same length. This metric is primarily used in the context of coding theory, but it can also be applied in 
        other fields to quantify dissimilarity between strings of the same length.

        :Funciton :
         
         This python function to calculate the Hamming distance between two strings of equal length
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "karolin"

        string2 = "kathrin"

        distance = hamming_distance(string1, string2)

        print(f"The Hamming distance between '{string1}' and '{string2}' is {distance}.")

        >>The Hamming distance between 'karolin' and 'kathrin' is 2

        """
        if len(str1) != len(str2):
            raise ValueError("Strings must be of equal length")
        distance = 0
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                distance += 1
                print(str1[i],str2[i])
        return distance
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def levenshtein_distance(str1, str2):
        """
        :Definition: 
        
        The Levenshtein distance, also known as edit distance, measures the minimum number of singlecharacter edits
        (insertions, deletions, or substitutions) required to change one string into another. 
        Named after the Soviet mathematician Vladimir Levenshtein, this metric is used to quantify the similarity 
        or dissimilarity between two strings regardless of their lengths.

        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "kitten"

        string2 = "sitting"

        distance = levenshtein_distance(string1, string2)
        
        print(f"The Levenshtein distance between '{string1}' and '{string2}' is {distance}.")

        >>The Levenshtein distance between 'kitten' and 'sitting' is 3.
 
        """
        len_str1 = len(str1)
        len_str2 = len(str2)
# Create a matrix to store the distances between substrings
        matrix = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
# Initialize the matrix with values from 0 to len_str1 for the first column
        for i in range(len_str1 + 1):
            matrix[i][0] = i
# Initialize the matrix with values from 0 to len_str2 for the first row
        for j in range(len_str2 + 1):
            matrix[0][j] = j
# Fill the matrix using dynamic programming
        for i in range(1, len_str1 + 1):
            for j in range(1, len_str2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                matrix[i][j] = min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + cost
                )
        return matrix[len_str1][len_str2]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    def damerau_levenshtein_distance(str1, str2):
        """
        :Definition:
         
        The Damerau-Levenshtein distance is an extension of the classic Levenshtein distance. It measures the 
        minimum number of operations (insertions, deletions, substitutions, or transpositions of adjacent 
        characters) needed to change one string into another. Transpositions involve swapping two adjacent 
        characters in a string. This distance metric accounts for these transpositions in addition to the basic edit 
        operations considered in the Levenshtein distance.

        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "sunday"

        string2 = "saturday"

        distance = damerau_levenshtein_distance(string1, string2)
       
         print(f"The Damerau-Levenshtein distance between '{string1}' and '{string2}' is {distance}.")

        >>The Damerau-Levenshtein distance between 'sunday' and 'saturday' is 2.
        
        """
        len_str1 = len(str1)
        len_str2 = len(str2)
# Create a matrix to store the distances between substrings
        matrix = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
# Initialize the matrix with values from 0 to len_str1 for the first column
        for i in range(len_str1 + 1):
            matrix[i][0] = i
# Initialize the matrix with values from 0 to len_str2 for the first row
        for j in range(len_str2 + 1):
            matrix[0][j] = j
# Fill the matrix using dynamic programming
        for i in range(1, len_str1 + 1):
            for j in range(1, len_str2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
# Calculate transposition cost
                transposition_cost = 1
                if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                    transposition_cost = 0
                matrix[i][j] = min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + cost,
                        matrix[i - 2][j - 2] + transposition_cost # Transposition
                )
        return matrix[len_str1][len_str2]
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def jaro_distance(str1, str2):
        """
        :Definition: 
        
        The Jaro distance is a measure of similarity between two strings. It calculates the similarity between 
        strings by measuring the number of matching characters divided by the total number of characters in the 
        two strings, taking into account the transpositions of characters. This metric provides a value between 0 
        (no similarity) and 1 (perfect match).

        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "Dwayne"

        string2 = "Duane"

        distance = jaro_distance(string1, string2)

        print(f"The Jaro distance between '{string1}' and '{string2}' is {distance}.")

        >>The Jaro distance between 'Dwayne' and 'Duane' is 0.8222222222222223.

        """
# Length of strings
        len_str1 = len(str1)
        len_str2 = len(str2)
# Matching distance (maximum number of characters to match)
        match_distance = max(len_str1, len_str2) // 2 - 1
        if match_distance < 0:
            match_distance = 0
# Arrays to store matching characters
        str1_matches = [False] * len_str1
        str2_matches = [False] * len_str2
# Count of matching characters
        matches = 0
# Count of transpositions
        transpositions = 0
# Find matching characters
        for i in range(len_str1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len_str2)
            
            for j in range(start, end):
                if not str2_matches[j] and str1[i] == str2[j]:
                    str1_matches[i] = True
                    str2_matches[j] = True
                    matches += 1
                    break
# If there are no matches, return 0
        if matches == 0:
            return 0.0
# Count transpositions
        k = 0
        for i in range(len_str1):
            if str1_matches[i]:
                while not str2_matches[k]:
                    k += 1
                if str1[i] != str2[k]:
                    transpositions += 1
                k += 1

        transpositions //= 2 # Divide by 2 as transpositions were counted twice
# Calculate Jaro distance
        similarity = (
            matches / len_str1 +
            matches / len_str2 +
            (matches - transpositions) / matches
        ) / 3
        return similarity
    

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def jaro_winkler_distance(str1, str2, prefix_scale=0.1):
        """
        :Definition :

        The Jaro-Winkler distance is an extension of the Jaro distance metric that incorporates a prefix scale 
        factor to give more weight to matching characters at the beginning of the strings. This modified metric 
        increases the similarity score when the initial characters of two strings match, providing a higher 
        similarity measure for strings that share a common prefix. It's particularly useful for measuring the 
        similarity between strings with slight differences, such as typos or variations in names.
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "Dwayne"
        
        string2 = "Duane"
        
        distance = jaro_winkler_distance(string1, string2)
        
        print(f"The Jaro-Winkler distance between '{string1}' and '{string2}' is {distance}.")

        >>The Jaro-Winkler distance between 'Dwayne' and 'Duane' is 0.8400000000000001.

        
        """
# Jaro similarity
        jaro_similarity = jaro_distance(str1, str2)
# Length of common prefix (up to a maximum of 4 characters)
        prefix_length = 0
        max_prefix_length = min(4, min(len(str1), len(str2)))
        for i in range(max_prefix_length):
            if str1[i] == str2[i]:
                prefix_length += 1
            else:
                break
# Calculate Jaro-Winkler distance
        jaro_winkler_distance = jaro_similarity + (prefix_length * prefix_scale * (1 - jaro_similarity))
        return jaro_winkler_distance
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def longest_common_subsequence(str1, str2):
        """
        :Definition:

        The Longest Common Subsequence (LCS) refers to the longest sequence of characters (or elements) that 
        are present in the same order in two or more sequences. This doesn't require consecutive positions 
        within the original sequences. LCS is a dynamic programming problem commonly used in bioinformatics, 
        computer science, and data comparison scenarios to find similarities between sequences
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        # Example usage
        
        sequence1 = "AGGTAB"
        
        sequence2 = "GXTXAYB"
        
        result = longest_common_subsequence(sequence1, sequence2)
        
        print("Longest Common Subsequence:", result)

        >>Longest Common Subsequence: GTAB

        
        """
        m = len(str1)
        n = len(str2)
# Initializing a table to store the lengths of LCS
        lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
# Building the LCS table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
# Finding the longest common subsequence
        lcs = ""
        i, j = m, n
        while i > 0 and j > 0:
            if str1[i - 1] == str2[j - 1]:
                lcs = str1[i - 1] + lcs
                i -= 1
                j -= 1
            elif lcs_table[i - 1][j] > lcs_table[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return lcs    

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def longest_common_substring(str1, str2):
        """
        :Definition:

        The Longest Common Substring (LCS) problem aims to find the longest string that is a substring of two or 
        more strings.

        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        sequence1 = "ABCDEF"
        
        sequence2 = "ZXCDEYF"
        
        result = longest_common_substring(sequence1, sequence2)
        
        print("Longest Common Substring:", result)

        >>Longest Common Substring: CDE

        """
        
        m = len(str1)
        n = len(str2)
# Initializing a table to store lengths of longest common suffixes
        lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
# Variable to store the length of the longest common substring
        max_length = 0
# Variable to store the ending index of the longest common substring
        end_index = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                    if lcs_table[i][j] > max_length:
                        max_length = lcs_table[i][j]
                        end_index = i
                else:
                    lcs_table[i][j] = 0
        start_index = end_index - max_length
        return str1[start_index:end_index]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  
    import nltk
    def qgram_distance(s1, s2, q):
        """
        :Definition:

        Q-gram distance, also known as q-gram similarity, measures the similarity between two strings by 
        considering sequences of q consecutive characters (or tokens) within the strings. It counts the number of 
        common q-grams between the strings and computes a similarity score based on these shared q-grams. 
        This distance metric is useful in various text processing tasks, including plagiarism detection, spell 
        correction, and information retrieval.
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "hello"
        
        string2 = "hallo"
        
        q_value = 2
        
        distance = qgram_distance(string1, string2, q_value)
        
        print(f"The {q_value}-gram distance between '{string1}' and '{string2}' is: {distance}")
        
        >>The 2-gram distance between 'hello' and 'hallo' is: 0.4

        """
        qgrams1 = nltk.ngrams(s1, q)
        qgrams2 = nltk.ngrams(s2, q)
        set1 = set(qgrams1)
        set2 = set(qgrams2)
        return 1 - len(set1.intersection(set2)) / len(set1.union(set2))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def dice_coefficient(s1, s2, n):
        """
        :Definition :

        The overlap coefficient measures the similarity of two sets by calculating the ratio of the intersection to 
        the smaller of the two sets. In the context of strings, it can be used in conjunction with the edit distance.


        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "hello"
        
        string2 = "hallo"
        
        n_value = 2
        
        dice_coeff = dice_coefficient(string1, string2, n_value)
        
        >>0.8
        
        """
# Tokenize strings into n-grams
        s1_ngrams = set(nltk.ngrams(s1, n))
        s2_ngrams = set(nltk.ngrams(s2, n))
# Calculate intersection
        intersection = len(s1_ngrams.intersection(s2_ngrams))
# Compute Dice coefficient
        dice_coeff = (2.0 * intersection) / (len(s1_ngrams) + len(s2_ngrams))
        return dice_coeff
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def jaccard_similarity(s1, s2, n):
        """
        :Defintion : 

        Jaccard similarity measures the similarity between two sets by comparing their intersection to their 
        union. In the context of strings and edit distance, it can be used as a similarity measure after 
        transforming strings into sets of n-grams.
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        string1 = "hello"
        
        string2 = "hallo"
        
        n_value = 2
        
        jaccard_sim = jaccard_similarity(string1, string2, n_value)
        
        >>0.33
        """
# Tokenize strings into n-grams
        s1_ngrams = set(nltk.ngrams(s1, n))
        s2_ngrams = set(nltk.ngrams(s2, n))
# Calculate intersection and union
        intersection = len(s1_ngrams.intersection(s2_ngrams))
        union = len(s1_ngrams.union(s2_ngrams))
# Compute Jaccard similarity
        if union == 0:
            return 0 # To avoid division by zero
        return intersection / union

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    from collections import Counter
    def bag_distance(s1, s2):
        """
        :Definition:

        The Bag distance, also known as the Bag of Words distance, measures the difference between two sets 
        of elements, regardless of order or frequency. In the context of strings and edit distance, it's typically 
        used after transforming strings into sets of words.

        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:
        
        string1 = "hello world"
        
        string2 = "world hello"
        
        distance = bag_distance(string1, string2)
        
        >>0

        """
# Tokenize strings into words
        words1 = s1.split()
        words2 = s2.split()
# Create Counters for the words in each string
        counter1 = Counter(words1)
        counter2 = Counter(words2)
# Compute Bag distance as the difference in sets
        bag_diff = sum((counter1 - counter2).values()) + sum((counter2 - counter1).values())
        
        return bag_diff    
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def edit_distance_with_dice_coefficient(str1, str2):
        """
        :Definition:

        The Dice coefficient is another similarity measure that compares the similarity of two sets. In the context 
        of strings and edit distance, it can be utilized after converting strings into sets of n-grams.
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:
        
        """
        set1 = set(str1)
        set2 = set(str2)
        union_size = len(set1.union(set2))
        intersection_size = len(set1.intersection(set2))
        edit_distance = (union_size - intersection_size) / 2
        return edit_distance
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap_penalty=-1):
        """
        :Definition:

        Smith-Waterman is a dynamic programming algorithm used to find the optimal local alignment between 
        two sequences. While edit distance measures the minimum number of operations (insertions, deletions, 
        substitutions) needed to transform one sequence into another, Smith-Waterman specifically focuses on 
        finding the most similar subsequence(s) within those sequences. It's commonly used in bioinformatics 
        for sequence alignment tasks like DNA or protein comparisons, providing a more nuanced similarity 
        measurement by identifying local similarities even within larger sequences.

        :param seq1: first string
        :type seq1: string

        :param seq2 : seconde string
        :type seq2: string

        :param match : by default = 2

        :param mismatch : by default =1

        :Example:
        
        sequence1 = "AGCACACA"
        
        sequence2 = "ACACACTA"

        
        alignment1, alignment2 = smith_waterman(sequence1, sequence2)
        
        print("Sequence 1:", alignment1)
        
        print("Sequence 2:", alignment2)
        
        >>Sequence 1: AGCACAC-A
        
        >>Sequence 2: A-CACACTA
        """
        len_seq1 = len(seq1)
        len_seq2 = len(seq2)
# Initialize the score matrix with zeros
        score_matrix = [[0] * (len_seq2 + 1) for _ in range(len_seq1 + 1)]
# Fill the score matrix and find the maximum score
        max_score = 0
        max_i, max_j = 0, 0
        
        for i in range(1, len_seq1 + 1):
            for j in range(1, len_seq2 + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    score = score_matrix[i - 1][j - 1] + match
                else:
                    score = max(
                            score_matrix[i - 1][j] + gap_penalty,
                            score_matrix[i][j - 1] + gap_penalty,
                            score_matrix[i - 1][j - 1] + mismatch
                                )
                score_matrix[i][j] = max(0, score)
                
                if score_matrix[i][j] > max_score:
                    max_score = score_matrix[i][j]
                    max_i, max_j = i, j
# Traceback to find the aligned sequences
        aligned_seq1, aligned_seq2 = '', ''
        i, j = max_i, max_j
        while i > 0 and j > 0 and score_matrix[i][j] > 0:
            current_score = score_matrix[i][j]
            diagonal_score = score_matrix[i - 1][j - 1]
            up_score = score_matrix[i - 1][j]
            left_score = score_matrix[i][j - 1]
            
            
            if current_score == diagonal_score + match:
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                i -= 1
                j -= 1
            elif current_score == up_score + gap_penalty:
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = '-' + aligned_seq2
                i -= 1
            elif current_score == left_score + gap_penalty:
                aligned_seq1 = '-' + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                j -= 1
        return max_score, aligned_seq1, aligned_seq2

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    import jellyfish

    def edit_distance_with_editex(str1, str2):
        """
        :Definition:

        The Editex algorithm is a variation of the edit distance calculation that aims to consider phonetic 
        similarities between words by assigning weights to different types of phonetic transformations (like 
        phoneme insertions, deletions, substitutions, etc.). It's particularly useful in spelling correction or 
        approximate string matching tasks where phonetic similarity matters.
        Implementing Editex involves assigning specific weights to phonetic transformations and then using 
        dynamic programming techniques to calculate the distance between two strings based on these 
        weighted transformations. The algorithm adjusts the weights according to phonetic similarity to improve 
        the accuracy of the distance measurement.

        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:

        str1 = "kitten"
        
        str2 = "sitting"
        
        result = editex_distance(str1, str2)
        
        print(f"Editex distance between '{str1}' and '{str2}': {result}")

        >>Editex distance between 'kitten' and 'sitting': 3

        """
        m, n = len(str1), len(str2)

    # matrix for save information of scops:
        dp = [[0] * (n + 1) for _ in range(m + 1)]

    # filling matrix
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],        # remove
                                    dp[i - 1][j],        # insert
                                    dp[i - 1][j - 1])    # swap

        return dp[m][n]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
    

    def align_syllables(str1, str2):
        """
        :Definition:

        
        Aligning syllables within the context of edit distance involves segmenting words into their constituent 
        syllables and then performing an alignment based on these syllables. This process helps in understanding 
        the similarity between words at a finer level by considering their syllabic structure.
        Implementing syllable alignment in edit distance would entail a few steps:
        
            1. **Syllable Segmentation:** Utilize a syllable segmentation algorithm to break down words into their 
        constituent syllables. You might use linguistic rules, dictionaries, or machine learning models designed 
        for syllable segmentation.

            2. **Alignment:** Apply an alignment algorithm (like dynamic programming used in edit distance) 
        considering the syllables instead of individual characters. This aligns the syllables between two words, 
        allowing for more nuanced comparison and calculation of similarity.

            3. **Scoring:** Define scoring mechanisms that consider syllable similarity, insertion, deletion, and 
        substitution while aligning the syllables. These scores would influence the alignment and distance 
        calculation.
        
        Implementing this process in Python involves integrating a syllable segmentation algorithm (which could 
        vary based on linguistic rules or external libraries) with an alignment algorithm to handle syllabic units 
        rather than individual characters.
        The specific implementation might vary based on the chosen syllable segmentation approach and the 
        alignment algorithm employed to consider syllables instead of characters. Integrating these two aspects 
        is crucial to perform syllable-level alignment within the context of edit distance.
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:
        
        str1 = "editing"
        
        str2 = "edting"
        
        result = syllable_alignment_edit_distance(str1, str2)
        
        print(f"Syllable Alignment Edit Distance between '{str1}' and '{str2}': {result}")
        
        >>Syllable Alignment Edit Distance between 'editing' and 'edting': 4
         
        """
        # syllabify algorithm :
        def syllabify(text):
            syllables = []
            current_syllable = ""
            vowels = "aeiouyAEIOUY"
            for char in text:
                if char in vowels:
                    current_syllable += char
                else:
                    if current_syllable:
                        syllables.append(current_syllable)
                        current_syllable = ""
                    syllables.append(char)
            if current_syllable:
                syllables.append(current_syllable)
            return syllables

    # divide to syllables:
        syllables1 = syllabify(str1)
        syllables2 = syllabify(str2)

    # compute edit distance between strings : 
        edit_distance = 0
        for syllable1, syllable2 in zip(syllables1, syllables2):
    
            edit_distance += Edit_Distance.levenshtein_distance(syllable1, syllable2)

        return edit_distance
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    def geshtalt_distance(str1, str2):
        """
        :Definition:

        Gesalt means "shape" or "pattern" in German. This concept goes back to the theory of Gesalt Psychology, which enters the field of experimental psychology. In the field of pattern recognition, we seek to recognize shapes, forms and patterns in data.

        How it works:
        Data Preprocessing:
            First, the input data (image or signal) is preprocessed according to the desired subject. Operations such as color conversion, contrast enhancement, or filtering may be applied.

        Feature extraction:
            From the pre-processed data, important and meaningful features are extracted. These features can include key points, dark or light areas, or specific sub-data features that the Gesalt Pattern Matching algorithm looks for.

        Comparison with models:
            The Gesalt Pattern Matching algorithm compares the extracted data with the patterns using its models and patterns. These patterns may be default or pre-defined.

        Decision making and diagnosis:
            Based on the comparison and the information obtained from the previous steps, the algorithm decides whether the desired pattern exists in the data or not. This step can lead to the final diagnosis based on matching the patterns in the input data.

        Justification of some algorithms:
            Gesalt Pattern Matching algorithms typically use machine learning theories, neural networks, or more sophisticated methods to detect patterns and shapes in data.
            These algorithms can be used in medical imaging, face recognition, or other fields. But for each mode, they usually need special settings and configurations to work in the best possible way.
        
        :param str1: first string.
        :type str1: string
        
        :param str2: seconde string.
        :type str2: string

        :Example:
       
        str1 = "kitten"
        
        str2 = "sitting"
        
        result = levenshtein_distance(str1, str2)
        
        print(f"Levenshtein Distance between '{str1}' and '{str2}': {result}")
        
        >>Levenshtein Distance between 'kitten' and 'sitting': 3
        
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

        return dp[m][n]

