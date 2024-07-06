library(BayesPrism)
library(ggplot2)
library(gridExtra)
library(ggpubr)
library(dplyr)

bp.res <- run.prism(prism = myPrism, n.cores=50)
bp.res
slotNames(bp.res)

theta <- get.fraction(bp=bp.res,
                       which.theta="final",
                       state.or.type="type")
 
head(theta)

theta.cv <- bp.res@posterior.theta_f@theta.cv
head(theta.cv)
 
Z.Myeloid <- get.exp (bp=bp.res,
                    state.or.type="type",
                    cell.name="Myeloid")