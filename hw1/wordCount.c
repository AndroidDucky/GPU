#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "wordCount.h"

//prints out to a text file after a countSort
void countSortPrint(const wordStruct *words)
{
    int i;
    FILE * fp;
    fp = fopen("sortedOccur.txt", "w+");
    fputs("|-----------------------|\n",fp);
    fprintf(fp,"|%-15s |%s\n","English Word", "Count |");
    fprintf(fp,"|-----------------------|\n");


    i = 0;
    while (words[i].word)
    {
        fprintf(fp,"|%-15s |  %d   |\n", words[i].word, words[i].count);
        fprintf(fp,"|-----------------------|\n");
        i++;
    }
    fprintf(fp,"I didn't do the extra credit\n");

    fclose(fp);
}
//prints out to a text file after a wordSort
void wordSortPrint(const wordStruct *words)
{
    int i;
    FILE * fp;
    fp = fopen("sortedWord.txt", "w+");
    fputs("|-----------------------|\n",fp);
    fprintf(fp,"|%-15s |%s\n","English Word", "Count |");
    fprintf(fp,"|-----------------------|\n");


    i = 0;
    while (words[i].word)
    {
        fprintf(fp,"|%-15s |  %d   |\n", words[i].word, words[i].count);
        fprintf(fp,"|-----------------------|\n");
        i++;
    }
    fprintf(fp,"I didn't do the extra credit\n");
    fclose(fp);
}


//adds a unique word to the list and if it isn't unique it adds a count to the word
int addUnique(const char *w, wordStruct *words)
{
    int i;
    i = 0;
    while (i < MAX_WORDS && words[i].word != NULL)
    {
        if (!strcmp(words[i].word, w))
        {
            words[i].count++;
            return (0);
        }
        i++;
    }
    if (i < MAX_WORDS)
    {
        words[i].word = strdup(w);
        words[i].count = 1;
        return (i + 1);
    }
    fprintf(stderr, "Not enough space in the array of words\n");
    exit(1);
}

//Formats the word to get pass through checking if its unique or not
void cleanWords(char *word)
{
    int i;

    i = 0;
    while (word[i])
    {
        word[i] = tolower(word[i]);
        if (!isalpha(word[i]))
        {
            removeSpecialCharacter(word, i);
            i--;
        }
        i++;
    }
}

//compares counts
int countCmp(const wordStruct *w1, const wordStruct *w2)
{
    if (w1->count > w2->count)
    {
        return (-1);
    }
    if (w1->count < w2->count)
    {
        return (1);
    }
    return (0);
}

//compares words
int wordCmp(const wordStruct *w1, const wordStruct *w2)
{
    if (strcmp(w1->word,w2->word)<1)
    {
        return (-1);
    }
    if (strcmp(w1->word,w2->word)>-1)
    {
        return (1);
    }
    return (0);
}


//Removes any special character
void removeSpecialCharacter(char *w, const int pos)
{
    char tmp[MAX_SIZE];

    strcpy(tmp, w);
    strcpy(w + pos, tmp + pos + 1);
}

//initializes the array of structs
void initWords(wordStruct *words)
{
    int i;

    for (i = 0; i < MAX_WORDS; i++)
    {
        words[i].word;
        words[i].count = 0;
    }
}
