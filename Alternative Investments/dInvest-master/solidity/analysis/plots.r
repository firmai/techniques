plot(blacklist$V3, blacklist$V2, xlab = "No of user criteria", ylab = "Gas consumed")
lo <- loess(blacklist$V2~blacklist$V3)
lines(predict(lo), col='red', lwd=2)

plot(users$V1, users$V2, xlab = "No of users", ylab = "Gas consumed")
lo <- loess(users$V2~users$V1)
lines(predict(lo), col='red', lwd=2)

