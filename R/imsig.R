if( !require(devtools) ) install.packages("devtools")
devtools::install_github( "ajitjohnson/imsig", INSTALL_opts = "--no-multiarch")

# Load the package
library("imsig")
load('exp.Rdata')

# Run
gene_stat (exp = exp, r = 0.6) 
pro = imsig(exp = exp, r = 0.6)  

# plot
plot_abundance (exp = exp, r = 0.6)
plot_network (exp = exp, r = 0.6)