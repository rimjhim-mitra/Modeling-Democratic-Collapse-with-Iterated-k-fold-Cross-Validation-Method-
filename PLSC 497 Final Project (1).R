# Clear environment
rm(list = ls())

#load libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(foreach)
library(verification)
library(PRROC)
library(pROC)
library(rcompanion)

#read and format data
demcollapse <- read_csv("demcollapsedata.csv")
dem_2020 = demcollapse[demcollapse$year==2020,]
nvc <-subset(demcollapse, year<2020)
dem_2020 = dem_2020 %>%
  subset(select = -oilgasrentsgdp)
nvc <- nvc[!is.na(nvc$demcollapse), ]

#checking range for variables + transformations
range(demcollapse$partyage, na.rm = TRUE)
demcollapse$partyage = transformTukey(demcollapse$partyage)
range(demcollapse$persparty, na.rm = TRUE)
range(demcollapse$v2paseatshare, na.rm = TRUE)
demcollapse$v2paseatshare = transformTukey(demcollapse$v2paseatshare)
range(demcollapse$v2exl_legitideol, na.rm = TRUE)
demcollapse$v2exl_legitideol = transformTukey(demcollapse$v2exl_legitideol + 2.7)
range(demcollapse$v2exl_legitlead, na.rm = TRUE)
demcollapse$v2exl_legitideol = transformTukey(demcollapse$v2exl_legitideol + 3.3)
range(demcollapse$v2exl_legitperf, na.rm = TRUE)
demcollapse$v2exl_legitperf = transformTukey(demcollapse$v2exl_legitperf + 2.4)
range(demcollapse$gwf_duration, na.rm = TRUE)
demcollapse$gwf_duration = transformTukey(demcollapse$gwf_duration)
range(demcollapse$gdppc, na.rm = TRUE)
demcollapse$gdppc = transformTukey(demcollapse$gdppc)
range(demcollapse$l12gr, na.rm = TRUE)
demcollapse$l12gr = transformTukey(demcollapse$l12gr + 24)
range(demcollapse$pres, na.rm = TRUE)
range(demcollapse$ivdem, na.rm = TRUE)
demcollapse$ivdem = transformTukey(demcollapse$ivdem)
range(demcollapse$leadertimeinpower, na.rm = TRUE)
demcollapse$leadertimeinpower = transformTukey(demcollapse$leadertimeinpower)
range(demcollapse$dem1990, na.rm = TRUE)
demcollapse$dem1990 = transformTukey(demcollapse$dem1990)
range(demcollapse$iv2xps_party, na.rm = TRUE)
demcollapse$iv2xps_party = transformTukey(demcollapse$iv2xps_party)
range(demcollapse$ipolarization, na.rm = TRUE)
demcollapse$ipolarization = transformTukey(demcollapse$ipolarization + 4)
range(demcollapse$iv2juhcind, na.rm = TRUE)
demcollapse$iv2juhcind = transformTukey(demcollapse$iv2juhcind + 2.1)
range(demcollapse$iv2x_libdem, na.rm = TRUE)
demcollapse$iv2x_libdem = transformTukey(demcollapse$iv2x_libdem)
range(demcollapse$iv2x_partipdem, na.rm = TRUE)
demcollapse$iv2x_partipdem = transformTukey(demcollapse$iv2x_partipdem)
range(demcollapse$iv2xpa_illiberal, na.rm = TRUE)
demcollapse$iv2xpa_illiberal = transformTukey(demcollapse$iv2xpa_illiberal)
range(demcollapse$iv2xpa_popul, na.rm = TRUE)
demcollapse$iv2xpa_popul = transformTukey(demcollapse$iv2xpa_popul)
range(demcollapse$iv2paclient, na.rm = TRUE)
demcollapse$iv2paclient = transformTukey(demcollapse$iv2paclient + 3)
range(demcollapse$iv2paind, na.rm = TRUE)
demcollapse$iv2paind = transformTukey(demcollapse$iv2paind + 3)
range(demcollapse$iv2pariglef, na.rm = TRUE)
demcollapse$iv2pariglef = transformTukey(demcollapse$iv2pariglef + 3)
range(demcollapse$iv2paseat, na.rm = TRUE)
demcollapse$iv2paseat = transformTukey(demcollapse$iv2paseat)



#calculate the baseline probability of the outcome
mean_demcollapse <- mean(nvc$demcollapse, na.rm = TRUE)

#plot the distribution of the outcome
ggplot(nvc, aes(x=demcollapse)) + 
  geom_bar() +
  labs(title = "Frequency of Demcollapse", x = "Democracy Collapse", y = "Frequency")


# initialize empty lists to store models and predictions
models <- list()
predictions <- list()

# set seed to reproduce results that rely on random number generators
set.seed(42)
# obtain 10 random numbers between 2 and 10000 to use as seed values for each loop/repeat
iseed <- sample(2:10000,10, replace=FALSE) 

#k-fold function
kayfold<-function(mymodel.f) {
  # Get predictor names for this model
  predictor.names <- all.vars(formula(mymodel.f))  # get predictor names
  predictor.names<-c("country", "year", predictor.names) # add country & year to predictor names list
  nvc.mymodel <- nvc[, c(predictor.names)]  # keep only model predictor variables + outcome variable

  # Keep only rows with no missing data
  rows.with.missing.data <- which(!complete.cases(nvc.mymodel))  # identify rows with missing data
  nvc.mymodel<- nvc.mymodel[-rows.with.missing.data, ]  # keep only rows with no missing data
  num.rows <- nrow(nvc.mymodel)

  # Data frame to store iterations of predicted probabilities and predicted class
  predictor.names<-c("country", "year", "demcollapse") # add country & year to predictor names list
  nvc.results<-nvc.mymodel[, c(predictor.names)]  # keep only outcome variable
  
  # set counter j to index iterations/repeats of the k-fold cross-validation
  j<-1    
  foreach(s = iseed) %do% {
    # For each iteration, manipulate a "temp" dataframe 
    nvc.temp<-nvc.mymodel
  
    # Randomly splits data into 5 folds
    set.seed(s)
    nvc.temp$index <- runif(num.rows) # add column for random number between 0,1
    folds <- cut(nvc.temp$index, breaks = 5, labels = FALSE)  # split data into five folds

    # Iterate over each value of 1:5 (for 5-fold cross-validation)
    foreach(k = 1:5) %do% {
      # Subset nvc.temp based on the kth fold of the data
      test.data <- nvc.temp[folds == k, ]
      # Subset nvc.mymodel based on the k-1 folds of the data
      train.data <- nvc.temp[folds != k, ]
      # Fit a logistic regression model to the train subset
      models[[k]] <- glm(mymodel.f, data = train.data, family = binomial)
      # Predicted values: betas from train.data model using X values from test.data 
      predictions[[k]] <- predict(models[[k]], newdata = test.data, type = "response")
      # Add predictions back into test.data
      test.data$pr <- predictions[[k]]
      cnames<-c("country", "year", "pr") # column names list
      test.data <- test.data[, c(cnames)]  # keep only four columns of data to merge back into nvc.mymodel
      cnames <- c("country", "year", paste0("pr", k))
      colnames(test.data) <- cnames
      nvc.temp<- merge(nvc.temp, test.data, by=c("country","year"), all=TRUE)
    }
  
    # Put each of the 5-fold predicted probabilities into a single column
    coln <- c("pr1", "pr2", "pr3", "pr4" , "pr5")
    nvc.pr <- nvc.temp[, coln]
    nvc.results[[paste0("pr", j)]] <- rowSums(nvc.pr, na.rm = TRUE)
  
    j<-j+1   # increase j value by 1 for the next repeat of the loop
  }
  
  # Average predicted probabilities from iterations
  coln <- c("pr1", "pr2", "pr3", "pr4", "pr5", "pr6", "pr7" , "pr8", "pr9", "pr10")
  nvc.results$pr<-rowMeans(nvc.results[,coln], na.rm = TRUE)  # mean predicted pr from each of iterations

  # Plot precision-recall curve
  pr <- pr.curve(nvc.results$demcollapse, nvc.results$pr, curve = TRUE )
  #plot(pr)
  print(pr$auc.integral)  # print the PR-AUC in the console output
  coln <- c("country", "year", "pr")
  out.nvc<-nvc.results[,coln]
  return(out.nvc)  # executing the function yields column of predicted probabilities

}

#models
model1.f <- formula(demcollapse ~ lpop + partyage + persparty + v2paseatshare + v2exl_legitideol + v2exl_legitlead 
                    + v2exl_legitperf + gwf_duration + gdppc + l12gr + priormil + pres + ivdem + leadertimeinpower 
                    + dem1990 + iv2xps_party + ipolarization + iv2juhcind + iv2x_libdem + iv2x_partipdem + iv2xpa_illiberal
                    + iv2xpa_popul + iv2paclient + iv2paind + iv2parelig + iv2pariglef + iv2paseat)
model2.f <- formula(demcollapse ~ lpop + gdppc + priormil + pres + ivdem + iv2xps_party)
model3.f <- formula(demcollapse ~ lpop + v2paseatshare + iv2paclient + iv2paind  + iv2parelig + iv2pariglef)
model4.f <- formula(demcollapse ~ lpop + dem1990 + gwf_duration + iv2x_libdem + iv2juhcind)
model5.f <- formula(demcollapse ~ lpop + l12gr + iv2xpa_illiberal + iv2xpa_popul + ipolarization)


#data frame to store predicted probabilities from each model
coln <- c("country", "year", "demcollapse")
nvc.final<-nvc[,coln]
 

# iterate over models and populate nvc.final
mnames <- c("model1", "model2", "model3", "model4", "model5")
for (m in mnames) {
  pr0 <- kayfold(get(paste0(m, ".f")))  # Assuming kayfold is a function returning vector of predicted probabilities
  nvc.final<-merge(nvc.final, pr0, by = c("country", "year"), all = TRUE)
  colnames(nvc.final)[which(colnames(nvc.final) == "pr")] <- paste0(m, "pr")
}
 


# sample forecast
predictor.names <- all.vars(formula(model4.f))
predictor.names <- c("country", "year", predictor.names) 
dem_2020 <- dem_2020[, predictor.names] 

best.model <- glm(model4.f, data = nvc, family = binomial)
dem_2020$pr <- predict(best.model, newdata = dem_2020, type = "response")

# Reorder the rows by value of pr
dem_2020 <- dem_2020[order(dem_2020$pr), ]
print(nrow(dem_2020))
print(max(dem_2020$pr))

common_xlim <- c(0, 0.25)
p1<-dotchart(dem_2020$pr[1:35], labels = dem_2020$country[1:35],
             cex = 0.55,  
             lwd=0.05, 
             lty=3, 
             pch=20,
             xlim=common_xlim,
             main = "Model forecasts for 2020",
             xlab = "Predicted Probability of Democracy Collapse",
             ylab = "Country",
)
dev.copy(pdf,'p1.pdf')
dev.off()

common_xlim <- c(0, 0.25)
p1<-dotchart(dem_2020$pr[36:69], labels = dem_2020$country[36:69],
             cex = 0.55,  
             lwd=0.05, 
             lty=3, 
             pch=20,
             xlim=common_xlim,
             main = "Model forecasts for 2020",
             xlab = "Predicted Probability of Democracy Collapse",
             ylab = "Country",
)
dev.copy(pdf,'p1.pdf')
dev.off()
