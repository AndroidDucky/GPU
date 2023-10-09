#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wordCount.h"

int main(int argc, char **argv)
{
    //declarations
    FILE *fp;
    wordStruct words[MAX_WORDS];
    int index;
    int NumOfWords;

    fp = fopen(argv[1], "r");
    //checking if the file is accessiable 
    if (fp == NULL)
    {
        printf("Unable to open file.\n");
        exit(EXIT_FAILURE);
    }
    NumOfWords = 0;
    initWords(words);
    char word[100];
    //going through the file and getting the words to add them to a list if they are unique or not
    while (fscanf(fp, "%s", word)==1)
    {
        if (strcmp(word, ",") > 0)
        {
            cleanWords(word);

            index = addUnique(word, words);
            if (index)
            {
                NumOfWords = index;
            }
        }
    }
    fclose(fp);
    qsort(words, NumOfWords, sizeof(wordStruct), (int (*)(const void *, const void *))wordCmp);
    wordSortPrint(words);
    qsort(words, NumOfWords, sizeof(wordStruct), (int (*)(const void *, const void *))countCmp);
    countSortPrint(words);

    return (0);
}