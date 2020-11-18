
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
        print(str(data))
        print(status(data))
        plot_num(data)
        
        temp <- status(data)$unique
       
}

basic_eda_r(df_1)

temp <- status(df_1)$unique
ifelse (temp < 10, lapply(df_1, function(x) unique(x)), NA)

freq(df_1)
