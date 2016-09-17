import matchTest

train_rots = [20]
#train_rots = [10, 20, 40, 80, 0]
#num_samples = [101, 301, 501, 701, 1001]
num_samples = [101, 501, 1001]
#angles = [10, 20, 40, 90, 180]
angles = [10, 40, 180]
#drop_ratio = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
drop_ratio = [0.2, 0.4, 0.6, 0.8, 1]

tests = []
i = 0
for ratio in drop_ratio:
#    with open("results.txt", "a") as myfile:
#        myfile.write("bunny_40_5_3_500_filter40.ckpt drop bad matches " + str(ratio))
#        myfile.close()
    for rot in train_rots:
        for nums in num_samples:
            for angle in angles:
                tests.append(["-1", str(rot), str(nums), str(angle), str(i), str(ratio)])
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
