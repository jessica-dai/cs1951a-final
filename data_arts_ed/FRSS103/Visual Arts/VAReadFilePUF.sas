FILENAME MyDat '<path to VAFRSS103PUF.dat>';
DATA VAFRSS103PUF;
   INFILE MyDat LRECL=917;
   INPUT
      @1    IDNUMBER         $5.
      @6    SIZE             best1.
      @7    URBAN            best1.
      @8    OEREG            best1.
      @9    MINST            best1.
      @10   POVST            best2.
      @12   LEVEL            best1.
      @13   Q1               best1.
      @14   Q2               best1.
      @15   T_Q3             best2.
      @17   Q5               best1.
      @18   Q6AA             best1.
      @19   Q6AB             best2.
      @21   T_Q6AC           best2.
      @23   T_Q6AD           best2.
      @25   Q6AE             best4.
      @29   Q6BA             best1.
      @30   Q6BB             best2.
      @32   T_Q6BC           best2.
      @34   T_Q6BD           best2.
      @36   Q6BE             best4.
      @40   Q6CA             best1.
      @41   Q6CB             best2.
      @43   T_Q6CC           best2.
      @45   T_Q6CD           best2.
      @47   Q6CE             best4.
      @51   Q6DA             best1.
      @52   Q6DB             best2.
      @54   T_Q6DC           best2.
      @56   T_Q6DD           best2.
      @58   Q6DE             best4.
      @62   Q6EA             best1.
      @63   Q6EB             best2.
      @65   T_Q6EC           best2.
      @67   T_Q6ED           best2.
      @69   Q6EE             best4.
      @73   Q6FA             best1.
      @74   Q6FB             best2.
      @76   T_Q6FC           best2.
      @78   T_Q6FD           best2.
      @80   Q6FE             best4.
      @84   Q7A              best2.
      @86   Q7B              best4.
      @90   Q7C              best2.
      @92   Q8               best1.
      @93   Q9               best1.
      @94   Q10AA            best1.
      @95   Q10AB            best2.
      @97   Q10BA            best1.
      @98   Q10BB            best2.
      @100  Q10CA            best1.
      @101  Q10CB            best2.
      @103  Q11AA            best1.
      @104  T_Q11AB          best4.
      @108  BACH_VISED       best1.
      @109  BACH_VISARTS     best1.
      @110  BACH_EDU         best1.
      @111  BACH_OTHER       best1.
      @112  Q11BA            best1.
      @113  T_Q11BB          best4.
      @117  Q11CA            best1.
      @118  GRAD_VISED       best1.
      @119  GRAD_VISARTS     best1.
      @120  GRAD_EDU         best1.
      @121  GRAD_OTHER       best1.
      @122  Q11DA            best1.
      @123  T_Q11DB          best2.
      @125  DEG1_INFIELD     best1.
      @126  Q12              best2.
      @128  T_Q13            best2.
      @130  Q14AA            best1.
      @131  Q14AB            best2.
      @133  Q14BA            best1.
      @134  Q14BB            best2.
      @136  Q14CA            best1.
      @137  Q14CB            best2.
      @139  Q14DA            best1.
      @140  Q14DB            best2.
      @142  Q14EA            best1.
      @143  Q14EB            best2.
      @145  Q14FA            best1.
      @146  Q14FB            best2.
      @148  Q14GA            best1.
      @149  Q14GB            best2.
      @151  Q14HA            best1.
      @152  Q14HB            best2.
      @154  Q15A             best1.
      @155  Q15B             best1.
      @156  Q15C             best1.
      @157  Q15D             best1.
      @158  Q15E             best1.
      @159  Q15F             best1.
      @160  Q15G             best1.
      @161  Q16A             best1.
      @162  Q16B             best1.
      @163  Q16C             best1.
      @164  Q16D             best1.
      @165  Q17A             best1.
      @166  Q17B             best1.
      @167  Q17C             best1.
      @168  Q17D             best1.
      @169  Q18A             best1.
      @170  Q18B             best1.
      @171  Q18C             best1.
      @172  Q18D             best1.
      @173  Q18E             best1.
      @174  Q18F             best1.
      @175  Q18G             best1.
      @176  Q18H             best1.
      @177  Q18I             best1.
      @178  Q18J             best1.
      @179  Q19A             best1.
      @180  Q19B             best1.
      @181  Q19C             best1.
      @182  Q19D             best1.
      @183  Q19E             best1.
      @184  Q19F             best1.
      @185  Q19G             best1.
      @186  Q19H             best1.
      @187  Q19I             best1.
      @188  Q20BOX           best1.
      @189  Q20A             best2.
      @191  Q20B             best2.
      @193  Q20C             best2.
      @195  Q20D             best2.
      @197  Q20E             best2.
      @199  Q20F             best2.
      @201  Q20G             best2.
      @203  Q21A             best1.
      @204  Q21B             best1.
      @205  Q21C             best1.
      @206  Q21D             best1.
      @207  Q21E             best1.
      @208  Q21F             best1.
      @209  Q21G             best1.
      @210  I_Q6AC           $1.
      @211  I_Q6AD           $1.
      @212  I_Q6AE           $1.
      @213  I_Q6BA           $1.
      @214  I_Q6BC           $1.
      @215  I_Q6BD           $1.
      @216  I_Q6BE           $1.
      @217  I_Q6CA           $1.
      @218  I_Q6CC           $1.
      @219  I_Q6CD           $1.
      @220  I_Q6CE           $1.
      @221  I_Q6DA           $1.
      @222  I_Q6DC           $1.
      @223  I_Q6DD           $1.
      @224  I_Q6DE           $1.
      @225  I_Q6EA           $1.
      @226  I_Q6EC           $1.
      @227  I_Q6ED           $1.
      @228  I_Q6EE           $1.
      @229  I_Q6FA           $1.
      @230  I_Q6FB           $1.
      @231  I_Q6FC           $1.
      @232  I_Q6FD           $1.
      @233  I_Q6FE           $1.
      @234  I_Q7A            $1.
      @235  I_Q7B            $1.
      @236  I_Q7C            $1.
      @237  I_Q9             $1.
      @238  I_Q10AB          $1.
      @239  I_Q10BB          $1.
      @240  I_Q10CB          $1.
      @241  I_Q11AA          $1.
      @242  I_Q11AB          $1.
      @243  I_Q11BB          $1.
      @244  I_Q11DB          $1.
      @245  I_Q12            $1.
      @246  I_Q13            $1.
      @247  I_Q14AA          $1.
      @248  I_Q14AB          $1.
      @249  I_Q14BA          $1.
      @250  I_Q14BB          $1.
      @251  I_Q14CA          $1.
      @252  I_Q14CB          $1.
      @253  I_Q14DA          $1.
      @254  I_Q14DB          $1.
      @255  I_Q14EA          $1.
      @256  I_Q14EB          $1.
      @257  I_Q14FA          $1.
      @258  I_Q14FB          $1.
      @259  I_Q14GA          $1.
      @260  I_Q14GB          $1.
      @261  I_Q14HA          $1.
      @262  I_Q14HB          $1.
      @263  I_Q15A           $1.
      @264  I_Q15C           $1.
      @265  I_Q15D           $1.
      @266  I_Q15E           $1.
      @267  I_Q15F           $1.
      @268  I_Q15G           $1.
      @269  I_Q17A           $1.
      @270  I_Q17B           $1.
      @271  I_Q17C           $1.
      @272  I_Q17D           $1.
      @273  I_Q18A           $1.
      @274  I_Q18B           $1.
      @275  I_Q18C           $1.
      @276  I_Q18D           $1.
      @277  I_Q18E           $1.
      @278  I_Q18F           $1.
      @279  I_Q18G           $1.
      @280  I_Q18H           $1.
      @281  I_Q18I           $1.
      @282  I_Q18J           $1.
      @283  I_Q19A           $1.
      @284  I_Q19B           $1.
      @285  I_Q19C           $1.
      @286  I_Q19D           $1.
      @287  I_Q19E           $1.
      @288  I_Q19F           $1.
      @289  I_Q19G           $1.
      @290  I_Q19H           $1.
      @291  I_Q19I           $1.
      @292  I_Q20A           $1.
      @293  I_Q20B           $1.
      @294  I_Q20C           $1.
      @295  I_Q20D           $1.
      @296  I_Q20E           $1.
      @297  I_Q20F           $1.
      @298  I_Q20G           $1.
      @299  I_Q21A           $1.
      @300  I_Q21B           $1.
      @301  I_Q21C           $1.
      @302  I_Q21D           $1.
      @303  I_Q21E           $1.
      @304  I_Q21F           $1.
      @305  I_Q21G           $1.
      @306  TFWT             best12.
      @318  TFWT1            best12.
      @330  TFWT2            best12.
      @342  TFWT3            best12.
      @354  TFWT4            best12.
      @366  TFWT5            best12.
      @378  TFWT6            best12.
      @390  TFWT7            best12.
      @402  TFWT8            best12.
      @414  TFWT9            best12.
      @426  TFWT10           best12.
      @438  TFWT11           best12.
      @450  TFWT12           best12.
      @462  TFWT13           best12.
      @474  TFWT14           best12.
      @486  TFWT15           best12.
      @498  TFWT16           best12.
      @510  TFWT17           best12.
      @522  TFWT18           best12.
      @534  TFWT19           best12.
      @546  TFWT20           best12.
      @558  TFWT21           best12.
      @570  TFWT22           best12.
      @582  TFWT23           best12.
      @594  TFWT24           best12.
      @606  TFWT25           best12.
      @618  TFWT26           best12.
      @630  TFWT27           best12.
      @642  TFWT28           best12.
      @654  TFWT29           best12.
      @666  TFWT30           best12.
      @678  TFWT31           best12.
      @690  TFWT32           best12.
      @702  TFWT33           best12.
      @714  TFWT34           best12.
      @726  TFWT35           best12.
      @738  TFWT36           best12.
      @750  TFWT37           best12.
      @762  TFWT38           best12.
      @774  TFWT39           best12.
      @786  TFWT40           best12.
      @798  TFWT41           best12.
      @810  TFWT42           best12.
      @822  TFWT43           best12.
      @834  TFWT44           best12.
      @846  TFWT45           best12.
      @858  TFWT46           best12.
      @870  TFWT47           best12.
      @882  TFWT48           best12.
      @894  TFWT49           best12.
      @906  TFWT50           best12.
      ;
RUN;
