//
// Created by amccullough5 on 1/20/23.
//

#ifndef LAB1_TIMING_H
#define LAB1_TIMING_H

#include <sys/time.h>



/* Subtract the `struct timeval' value 'then' from 'now',
   returning the difference as a float representing seconds
   elapsed.
*/
float elapsedTime(struct timeval now, struct timeval then);

double currentTime();

#endif //LAB1_TIMING_H
