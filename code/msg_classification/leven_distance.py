import numpy as np

class levenshtein_distance:
    def __init__(self, input_x, input_y):
        self.input_x = input_x
        self.input_y = input_y
	
    def le_dis(self):
        xlen = len(self.input_x) + 1 
        ylen = len(self.input_y) + 1
        dp = np.zeros(shape=(xlen, ylen), dtype=int)
        for i in range(0, xlen):
            dp[i][0] = i
        for j in range(0, ylen):
            dp[0][j] = j
        for i in range(1, xlen):
            for j in range(1, ylen):
                if self.input_x[i - 1] == self.input_y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[xlen - 1][ylen - 1]
    
    def leven_sim(self):
        leven_dis = self.le_dis()
        if leven_dis == 0:
			return 1.0
        else:
			return 1.0 / leven_dis
	
if __name__ == '__main__':
    ld = levenshtein_distance('abcd', 'abd')
    print(ld.le_dis()) # print out 1
    ld = levenshtein_distance('ace', 'abcd')
    print(ld.le_dis()) # print out 2
    ld = levenshtein_distance('hello world', 'hello')
    print(ld.le_dis()) # print out 4
    print(ld.leven_sim())
