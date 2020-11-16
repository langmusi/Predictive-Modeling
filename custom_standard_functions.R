
# groups in a dataframe and the size of the groups
# - the “exclude=NULL” option to display any missing or null values.
group_size <- function(my_val) {
        #if (class(my_val) == "")
        print(unique(my_val))
        print(table(my_val, exclude = NULL))  
}



##
## library: "funModeling", "tidyverse", "Hmisc"
library(funModeling); library(tidyverse); library("Hmisc")
basic_eda <- function(data){
        print(head(data))
        print(status(data))
        freq(data) 
        print(profiling_num(data))
        plot_num(data)
}

basic_eda_r <- function(data){
        print(head(data))
        print(status(data))
        plot_num(data)
}
