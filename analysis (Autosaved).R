getFileMean <- function (filename) {
	data <- read.csv(filename)
	dd <- subset(data, select = c(AL, Random, Batch))
	return(colMeans(dd))
}

