/********IMPACT*********/
/* this is the file that generates tables sent as response to QJE editor */
/* these tables do not appear in the paper */

**to be done after clean.impact.do, which is the last clean file in regressions6.do after the other clean.X.do files
macro define fullset_noGB "married edhi numhh unemployed age hh_inc hh_inc2"

set mem 200m
set more off

cap log close
log using "$directory/log files/SEED.QJEtable.impact.log", replace

/* Table 6: Impact */
/* 6 months */
use "$directory/dta files/seedanalysis_011204.dta", clear
xi: reg balchange treatment marketing, robust
*outreg using impact_ols, se replace coefastr 3aster title("Impact on Financial Savings", "Dependent Variable: Change in Total Institutional Savings Balance") ctitle("OLS")
xi: reg balchange treatment if (treatment == 1 | marketing == 1), robust
*outreg using impact_ols, se append coefastr 3aster ctitle("OLS; Commitment and Marketing Groups")

/* 12 months */
use "$directory/dta files/seedanalysis_011204_080404.dta", clear
xi: reg balchange treatment marketing, robust
*outreg using impact_ols, se append coefastr 3aster ctitle("OLS; All")

summ hh_inc if e(sample)==1
disp _b[treatment] / (_result(3)*10000)
**economic impact #2: % of prior balance
summ totbal if e(sample)==1
disp _b[treatment] / (_result(3)/100)

xi: reg balchange treatment if (treatment == 1 | marketing == 1), robust
*outreg using impact_ols, se append coefastr 3aster ctitle("OLS; Commitment and Marketing Groups")


xi: dprobit frac_change_00 treatment marketing, robust
*outreg using impact_ols, append coefastr 3aster title("Impact on Financial Savings", "Binary Outcome = 1 if Change in Balance > 0%")ctitle("OLS; Increase > 0%")
xi: dprobit frac_change_00 treatment if (treatment == 1 | marketing == 1), robust
*outreg using impact_ols, se append coefastr 3aster ctitle("OLS; Increase > 0%")
xi: dprobit frac_change_20 treatment marketing, robust
*outreg using impact_ols, se append coefastr 3aster ctitle("OLS; Increase > 20%")
xi: dprobit frac_change_20 treatment  if (treatment == 1 | marketing == 1), robust
*outreg using impact_ols, se append coefastr 3aster ctitle("OLS; Increase > 20%")

/*No longer in the tables--Editors want TOT out*/
/* 6 months */
use "$directory/dta files/seedanalysis_011204.dta", clear
ivreg balchange marketing (seedtakeup = treatment), robust
*outreg using impact_IV, se replace coefastr 3aster title("Impact on Financial Savings", "Dependent Variable: Change in Total Institutional Savings Balance") ctitle("IV")
ivreg balchange (seedtakeup = treatment) if (treatment == 1 | marketing == 1), robust
*outreg using impact_IV, se append coefastr 3aster ctitle("IV; Commitment and Marketing Groups")

/* 12 months */
use "$directory/dta files/seedanalysis_011204_080404.dta", clear
ivreg balchange marketing (seedtakeup = treatment), robust
*outreg using impact_IV, se append coefastr 3aster ctitle("IV; All")
ivreg balchange (seedtakeup = treatment) if (treatment == 1 | marketing == 1), robust
*outreg using impact_IV, se append coefastr 3aster ctitle("All; Commitment and Marketing Groups")

/*Calculation desired by editors*/
/*** Control Complier Mean (CCM) Stuff ***/
sum balchange if seedtakeup == 1
global complier =_result(3)
ivreg balchange marketing (seedtakeup = treatment), robust
global tot = _b[seedtakeup]
global ccm = $complier - $tot
display $tot - $ccm
display $tot/$ccm



/*** Table 7: quantile regressions ***/
/* 6 months */
use "$directory/dta files/seedanalysis_011204.dta", clear 	
sqreg balchange treatment marketing, q(.1 .2 .3 .4 .5 .6 .7 .8 .9)
*outreg using impact_sqreg, se replace coefastr 3aster title("Quantile Regression", "Dependent Variable: Balchange") ctitle("All")
sqreg balchange treatment if (treatment == 1 | marketing == 1), q(.1 .2 .3 .4 .5 .6 .7 .8 .9)
*outreg using impact_sqreg, se append coefastr 3aster ctitle("Treatment and Marketing Group")

/* 12 months */
use "$directory/dta files/seedanalysis_011204_080404.dta", clear
sqreg balchange treatment marketing, q(.1 .2 .3 .4 .5 .6 .7 .8 .9)
*outreg using impact_sqreg, se append coefastr 3aster ctitle("All")
sqreg balchange treatment if (treatment == 1 | marketing == 1), q(.1 .2 .3 .4 .5 .6 .7 .8 .9)
*outreg using impact_sqreg, se append coefastr 3aster ctitle("Treatment and Marketing Group")

/*** Table 8: Subgroups ***/
xi: reg balchange treatment marketing, robust
*outreg using impactsub_a, se replace coefastr 3aster title("Intent to Treat Effect of Subgroups", "Dependent Variable: Change in Total Institutional Savings Balance") ctitle("ALL")
xi: reg balchange treatment marketing female female_treat, robust 
*outreg using impactsub_a, se append coefastr 3aster ctitle("ALL")
xi: reg balchange treatment marketing active active_treat, robust
*outreg using impactsub_a, se append coefastr 3aster ctitle("ALL")
xi: reg balchange treatment marketing edhi edhi_treat, robust
*outreg using impactsub_a, se append coefastr 3aster ctitle("ALL")
xi: reg balchange treatment marketing hi_hh_inc hi_hh_inc_treat, robust 
*outreg using impactsub_a, se append coefastr 3aster ctitle("ALL")
xi: reg balchange treatment marketing hyper_mon_new2 hyper_mon_new2_treat, robust 
*outreg using impactsub_a, se append coefastr 3aster ctitle("ALL")
xi: reg balchange treatment marketing silly_mon_new2 silly_mon_new2_treat, robust 
*outreg using impactsub_a, se append coefastr 3aster ctitle("ALL")

/* impactsub_b not used in QJE tables, only impactsub_a */
xi: reg balchange treatment if (treatment == 1 | marketing == 1), robust 
*outreg using impactsub_b, se replace coefastr 3aster title("Intent to Treat Effect of Subgroups", "Dependent Variable: Change in Total Institutional Savings Balance") ctitle("Commitment and Marketing Treatment Groups")
xi: reg balchange treatment female female_treat if (treatment == 1 | marketing == 1), robust
*outreg using impactsub_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")
xi: reg balchange treatment active active_treat if (treatment == 1 | marketing == 1), robust
*outreg using impactsub_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")
xi: reg balchange treatment edhi edhi_treat if (treatment == 1 | marketing == 1), robust
*outreg using impactsub_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")
xi: reg balchange treatment hi_hh_inc hi_hh_inc_treat if (treatment == 1 | marketing == 1), robust
*outreg using impactsub_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")
xi: reg balchange treatment hyper_mon_new2 hyper_mon_new2_treat if (treatment == 1 | marketing == 1), robust
*outreg using impactsub_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")
xi: reg balchange treatment silly_mon_new2 silly_mon_new2_treat if (treatment == 1 | marketing == 1), robust
*outreg using impactsub_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")


/*** Test for New Savings ***/
reg nonseedbalchange treatment marketing, robust 
*outreg using newsav_a, se replace coefastr 3aster title("Tests for New Savings", "Dependent Variable: Change in total non-SEED Institutional Savings Balance") ctitle("ALL")
reg balchange treatment marketing, robust  
*outreg using newsav_a, se append coefastr 3aster ctitle("ALL")


/* newsav_b not used in QJW, only newsav_a */
reg nonseedbalchange treatment active if (treatment == 1 | marketing == 1)
*outreg using newsav_b, se replace coefastr 3aster title("Tests for New Savings", "Dependent Variable: Change in total non-SEED Institutional Savings Balance") ctitle("Commitment and Marketing Treatment Groups")
reg balchange treatment active if (treatment == 1 | marketing == 1) 
*outreg using newsav_b, se append coefastr 3aster ctitle("Commitment and Marketing Treatment Groups")

/*Appendix Table 2 */
xi: reg balchange treatment marketing, robust
*outreg using balchange_QJE, se replace coefastr 3aster title("Impact on Financial Savings", "Dependent Variable: Change in Total Institutional Savings Balance") ctitle("OLS")
xi: reg balchange treatment if (treatment == 1 | marketing == 1), robust
*outreg using balchange_QJE, se append coefastr 3aster ctitle("OLS; Commitment and Marketing Groups")
xi: reg balchange treatment marketing $fullset_noGB, robust  
*outreg using balchange_QJE, se append coefastr 3aster ctitle("OLS")
xi: reg balchange treatment $fullset_noGB if (treatment == 1 | marketing == 1), robust
*outreg using balchange_QJE, se append coefastr 3aster ctitle("OLS; Commitment and Marketing Groups")

/*these two regs taken from QJE response letter, where editors asked that we check robustness */
replace newtotbal = newtotbal/100
xi: reg newtotbal treatment marketing totbal, robust
*outreg using balchange_QJE, se append coefastr 3aster ctitle("OLS")
xi: reg newtotbal treatment totbal if (treatment == 1 | marketing == 1), robust
*outreg using balchange_QJE, se append coefastr 3aster ctitle("OLS; Commitment and Marketing Groups")


/** QJE response letter **/

use "$directory/dta files/seedanalysis_011204.dta", clear
generate post = 0
keep if call == 1
keep totbal newtotbal treatment marketing group married edhi numhh unemployed age hh_inc hh_inc2 post
save temp_011204.dta, replace
use "$directory/dta files/seedanalysis_080404.dta", clear
generate post = 1
keep if call == 1
keep totbal newtotbal treatment marketing group married edhi numhh unemployed age hh_inc hh_inc2 post
save temp_080404.dta, replace
append using temp_011204.dta
save temp_011204_080404.dta, replace

generate balance = totbal/100 if post == 0
replace balance = newtotbal/100 if post == 1
generate post_treatment = post*treatment
generate post_marketing = post*marketing

xi: reg balance post treatment marketing post_treatment post_marketing, robust
xi: reg balance post treatment post_treatment if (treatment == 1 | marketing == 1), robust

/*** Dfbeta ***/
*use seedanalysis_t2.dta, clear
use "$directory/dta files/seedanalysis_011204_080404.dta", clear

xi: reg balchange treatment marketing
dfbeta
*generate rDFtreatment = round(DFtreatment,0.01)
*generate rDFmarketing = round(DFmarketing,0.01)
*avplot treatment, mlabel(rDFtreatment)
*gsort- DFtreatment
*list DFtreatment in 1/20

/*testing role of dormant clients */
xi: dprobit frac_change_00 treatment marketing 
*outreg using impact_misc, replace coefastr 3aster title("Impact on Financial Savings", "Binary Outcome = 1 if Change in Balance > 0%")ctitle("OLS; Increase > 0%")
xi: dprobit frac_change_00 treatment active
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 0%")
xi: dprobit frac_change_20 treatment marketing 
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 20%")
xi: dprobit frac_change_20 treatment active
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 20%")

xi: dprobit frac_change_00 treatment marketing if active == 1
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 0%")
xi: dprobit frac_change_20 treatment marketing if active == 1
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 20%")
xi: dprobit frac_change_00 treatment marketing  if active == 0
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 0%")
xi: dprobit frac_change_20 treatment  marketing  if active == 0
*outreg using impact_misc, se append coefastr 3aster ctitle("OLS; Increase > 20%")

xi: reg newtotbal treatment marketing totbal, robust
*outreg using impact2_misc, se replace coefastr 3aster title("Impact on Financial Savings", "Dependent Variable: Change in Total Institutional Savings Balance") ctitle("OLS")
xi: reg newtotbal treatment totbal if (treatment == 1 | marketing == 1), robust
*outreg using impact2_misc, se append coefastr 3aster ctitle("OLS; Commitment and Marketing Groups")

/* survey non response */
/* randomized clinets have cid2 */
gen soughtsurvey = [group ~= ""]
gen gotsurvey = [call == 1 & soughtsurvey == 1]
bysort group: sum gotsurvey
gen randtreat = [group == "T"]
gen randmarket = [group == "M"]
gen randcontrol = [group == "C"]
reg gotsurvey randtreat randmarket if soughtsurvey == 1, robust
test (randtreat = randmarket = 0)
reg balchange randtreat randmarket if gotsurvey == 0 & soughtsurvey == 1 ,robust

egen dec1 = pctile(totbal), p(10)
egen dec2 = pctile(totbal), p(20)
egen dec3 = pctile(totbal), p(30)
egen dec4 = pctile(totbal), p(40)
egen dec5 = pctile(totbal), p(50)
egen dec6 = pctile(totbal), p(60)
egen dec7 = pctile(totbal), p(70)
egen dec8 = pctile(totbal), p(80)
egen dec9 = pctile(totbal), p(90)

gen increase = [balchange > 0]
*/
reg frac_change_00 treatment marketing if totbal <= dec1
reg frac_change_00 treatment marketing if totbal <= dec2 & totbal > dec1
reg frac_change_00 treatment marketing if totbal <= dec3 & totbal > dec2
reg frac_change_00 treatment marketing if totbal <= dec4 & totbal > dec3
reg frac_change_00 treatment marketing if totbal <= dec5 & totbal > dec4
reg frac_change_00 treatment marketing if totbal <= dec6 & totbal > dec5
reg frac_change_00 treatment marketing if totbal <= dec7 & totbal > dec6
reg frac_change_00 treatment marketing if totbal <= dec8 & totbal > dec7
reg frac_change_00 treatment marketing if totbal <= dec9 & totbal > dec8
reg frac_change_00 treatment marketing if totbal > dec9
