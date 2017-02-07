
# Reference :
# Marco Scutari (2010). Learning Bayesian Networks with the bnlearn R Package. 
# Journal of Statistical Software, 35(3), 1-22. 
# URL http://www.jstatsoft.org/v35/i03/.


# Model selection algorithms first try to learn the graphical structure of
# the Bayesian network (hence the name of structure learning algorithms) and then estimate
# the parameters of the local distribution functions conditional on the learned structure

library(bnlearn)
bn_df <- data.frame(coronary)

# Score-based algorithms: These algorithms assign a score to each candidate Bayesian
# network and try to maximize it with some heuristic search algorithm. Greedy search
# algorithms (such as hill-climbing or tabu search) are a common choice, but almost any
# kind of search procedure can be used.

# Learn the structure of a Bayesian network using a hill-climbing greedy search.
res <- hc(bn_df)
modelstring(res)
plot(res)

res$arcs <- res$arcs[-which((res$arcs[,'from'] == "M..Work" & res$arcs[,'to'] == "Family")),]
plot(res)

# Fit the parameters of a Bayesian network conditional on its structure
fittedbn <- bn.fit(res, data = bn_df)

print(fittedbn$Proteins)

# Note that both cpquery and cpdist are based on Monte Carlo particle filters, 
# and therefore they may return slightly different values on different runs.

numDraws <- 10000000
set.seed(1234)

cpquery(fittedbn, event = (Proteins=="<3"), evidence = ( Smoking=="no"), n = numDraws )

cpquery(fittedbn, event = (Proteins=="<3"), evidence = ( Smoking=="no" & Pressure==">140" ), n = numDraws )

cpquery(fittedbn, event = (Pressure==">140"), evidence = ( Proteins=="<3" ), n = numDraws )


# reasoning with the marks dataset

marks_df <- data.frame(marks)
res2 <- hc(marks_df)
modelstring(res2)
plot(res2)
marks_bn <- bn.fit(res2, data = marks_df)
cpquery(marks_bn, event = (ALG > 70), evidence = ( VECT > 90 & MECH > 70), n = numDraws )
cpquery(marks_bn, event = (ANL > 70), evidence = ( ALG > 80 & VECT < 40), n = numDraws )

# reasoning with the iris dataset

iris_df <- data.frame(iris)
res3 <- hc(iris_df)
modelstring(res3)
plot(res3)
iris_bn <- bn.fit(res3, data = iris_df)
cpquery(iris_bn, event = (Species == "setosa"), evidence = (Sepal.Length < 5 & Petal.Length > 1.4), n = numDraws)
predict (iris_bn, "Sepal.Length", iris_df[1,])
print(iris_bn$Sepal.Length)
