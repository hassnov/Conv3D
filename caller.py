import matchTest

train_rots = [40]
#train_rots = [10, 20, 40, 80, 0]
num_samples = [101, 301, 501, 701, 1001]
angles = [10, 20, 40, 90, 180]

tests = []
i = 0
for rot in train_rots:
    for nums in num_samples:
        for angle in angles:
            tests.append(["-1", str(rot), str(nums), str(angle), str(i)])
            i += 1

#for test in tests:
#  print test
#matchTest.main(tests[0])

start = 0
print len(tests)
for j in range(start, len(tests)):
  print 'test.........', tests[j]
  matchTest.main(tests[j])
#matchTest.main(['-1', '40', '101', '10'])
