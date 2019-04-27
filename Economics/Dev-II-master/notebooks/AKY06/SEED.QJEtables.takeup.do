*do SEEDdatasetup.QJE.do
***units for savings variables
**seedbal, balchange, balchange_t1, nonseedbalchange --in pesos
**totbal, newtotbal_t1, newtotbal--needs to be divided by 100

clear

set mem 200m
set more off

cap log close
log using "SEED.QJEtables.takeup.log", replace

use "seedanalysis_011204_080404_1.dta", clear
gen control = 1 if group == "C"
replace control = 0 if control ==.
macro define GB "GBloan GBloandefault"
macro define fullset "married edhi numhh unemployed age $GB hh_inc hh_inc2"
macro define fullset_noGB "married edhi numhh unemployed age hh_inc hh_inc2"

gen dormant_new = 1- active
drop dormant
rename dormant_new dormant
replace totbal = totbal/100
replace newtotbal = newtotbal/100
gen dist_GB = dbutuan if butuan ==1
replace dist_GB = dampayon if ampayon == 1
destring pop, ignore(",") replace
gen brgy_penetration = no_clients /pop
bysort brgy: egen sd_totbal = sd(totbal)
bysort brgy: egen mean_totbal = mean(totbal)

set more off
*log using table1_3.log, replace

*TABLE 1: Savings Goals
tab goal_category if seedtakeup==1
tab goal_type if seedtakeup==1
tab box if seedtakeup==1

*TABLE 2: Sumstats 
*Panel A
tabstat totbal active if call==1, by(group) stats(mean sem)
reg totbal treatment control if call ==1
reg active treatment control if call ==1
sort brgy
tabstat dist_GB brgy_penetration sd_totbal mean_totbal pop if call==1, by(group) stats(mean sem)
reg dist_GB treatment control if call ==1
reg brgy_penetration treatment control if call ==1
reg sd_totbal treatment control if call ==1
reg mean_totbal treatment control if call ==1
reg pop treatment control if call ==1

*Panel B
tabstat yearsed female age impatient_mon01 hyper_mon_new2 if call==1, by(group) stats(mean sem) 
reg yearsed treatment control if call ==1
reg female treatment control if call ==1
reg age treatment control if call ==1
reg impatient_mon01 treatment control if call ==1
reg hyper_mon_new2 treatment control if call ==1

*TABLE 3: Hypo Summary Table
tab impatient_mon01 impatient_mon67 if call==1

*log close

*TABLE 4: Determinants of Time Preference Questions
	*combines tables 9/10/11 hyper/impatient/silly
	*add share of household income (referee #1)
*TABLE 5: Determinants of Takeup
	*include female*HHpower + HHPower
	*include female*hyper (as well as splitting sample)

*TABLE 4: DETERMINANTS OF TIME PREFERENCE QUESTIONS

tab female if call==1 & fem_frac_veryown_inc !=., sum(impatient_mon01)
tab female if call==1 & fem_frac_veryown_inc !=., sum(hyper_mon_new2)
tab female if call==1 & fem_frac_veryown_inc !=., sum(silly_mon_any)

oprobit impatient_mon01 efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1, robust
*outreg using timepreferences, se replace coefastr 3aster title("Determinants of Responses to Time Preference Questions") ctitle("oprobit impatient_mon01, All")
oprobit impatient_mon01 efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1 & female ==1, robust
*outreg using timepreferences, se append coefastr 3aster ctitle("oprobit impatient_mon01, Female")
oprobit impatient_mon01 efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1 & female ==0, robust
*outreg using timepreferences, se append coefastr 3aster  ctitle("oprobit impatient_mon01, Male")

dprobit hyper_mon_new2 efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1, robust
*outreg using timepreferences, se append coefastr 3aster ctitle("dprobit hyper_mon_new2, All")
dprobit hyper_mon_new2 efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1 & female ==1, robust
*outreg using timepreferences, se append coefastr 3aster ctitle("dprobit hyper_mon_new2, Female")
dprobit hyper_mon_new2 efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1 & female ==0, robust
*outreg using timepreferences, se append coefastr 3aster  ctitle("dprobit hyper_mon_new2, Male")

dprobit silly_mon_any efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1, robust
*outreg using timepreferences, se append coefastr 3aster ctitle("dprobit silly_mon_any, All")
dprobit silly_mon_any efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1 & female ==1, robust
*outreg using timepreferences, se append coefastr 3aster ctitle("dprobit silly_mon_any, Female")
dprobit silly_mon_any efeel mis_efeel female fem_married $fullset_noGB lownow_high6mos_jj highnow_low6mos_jj  frac_veryown_inc fem_frac_veryown_inc active if call ==1 & female ==0, robust
*outreg using timepreferences, se append coefastr 3aster  ctitle("dprobit silly_mon_any, Male")


*TABLE 5: DETERMINANTS OF TAKEUP
gen female_hyper_mon_new2 = female * hyper_mon_new2
*	frac_veryown_inc fem_frac_veryown_inc 

#delimit ;
local apprep = "replace" ;

**COLUMN 1: ALL, NO FEMALE_HYPER INTERACTION;
#delimit ;
xi: dprobit seedtakeup hyper_mon_new2 female 
	impatient_200p250_01 impatient_250p200_01 impatient_200p250_67 impatient_250p200_67 
	fem_married $fullset fem_frac_veryown_inc_0_25 fem_frac_veryown_inc_25_50 fem_frac_veryown_inc_50_75 fem_frac_veryown_inc_75_100 frac_veryown_inc_0_25 frac_veryown_inc_25_50 frac_veryown_inc_50_75 frac_veryown_inc_75_100 
	active
	if group=="T" & call==1 & reached==1, robust ;
*capture `1' outreg using seedtakeup, se `apprep' coefastr 3aster title("Determinants of SEED Takeup") ctitle("All")
		addstat(Mean dependent variable, e(pbar)) ;


local apprep = "append";
**COLUMN 2: ALL, FEMALE_HYPER INTERACTION;
xi: dprobit seedtakeup hyper_mon_new2 female female_hyper_mon_new2 
	impatient_200p250_01 impatient_250p200_01 impatient_200p250_67 impatient_250p200_67 
	fem_married $fullset fem_frac_veryown_inc_0_25 fem_frac_veryown_inc_25_50 fem_frac_veryown_inc_50_75 fem_frac_veryown_inc_75_100 frac_veryown_inc_0_25 frac_veryown_inc_25_50 frac_veryown_inc_50_75 frac_veryown_inc_75_100 
	active
	if group=="T" & call==1 & reached==1, robust ;
*capture `1'*outreg using seedtakeup, se `apprep' coefastr 3aster title("Determinants of SEED Takeup") ctitle("All")
		addstat(Mean dependent variable, e(pbar)) ;

**COLUMN 3: FEMALE;
xi: dprobit seedtakeup hyper_mon_new2 female 
	impatient_200p250_01 impatient_250p200_01 impatient_200p250_67 impatient_250p200_67 
	fem_married $fullset frac_veryown_inc_0_25 frac_veryown_inc_25_50 frac_veryown_inc_50_75 frac_veryown_inc_75_100 
	active
	if group=="T" & call==1 & reached==1 & female==1, robust ;
*capture `1'*outreg using seedtakeup, se `apprep' coefastr 3aster title("Determinants of SEED Takeup") ctitle("Female")
		addstat(Mean dependent variable, e(pbar)) ;
**COLUMN 4: MALE;
xi: dprobit seedtakeup hyper_mon_new2 female 
	impatient_200p250_01 impatient_250p200_01 impatient_200p250_67 impatient_250p200_67 
	$fullset frac_veryown_inc_0_25 frac_veryown_inc_25_50 frac_veryown_inc_50_75 frac_veryown_inc_75_100 
	active
	if group=="T" & call==1 & reached==1 & female==0, robust ;
*capture `1'*outreg using seedtakeup, se `apprep' coefastr 3aster title("Determinants of SEED Takeup") ctitle("Male")
		addstat(Mean dependent variable, e(pbar)) ;

#delimit cr

/*
log on
*footnote 21
**ice cream: female
xi: dprobit seedtakeup hyper_ice_new2 female impatient_200p250_01ice impatient_250p200_01ice impatient_200p250_67ice impatient_250p200_67ice fem_married $fullset frac_veryown_inc_0_25 frac_veryown_inc_25_50 frac_veryown_inc_50_75 frac_veryown_inc_75_100 active if group=="T" & call==1 & reached==1 & female==1, robust
**rice: female
xi: dprobit seedtakeup hyper_rice_new2 female impatient_200p250_01rice impatient_250p200_01rice impatient_200p250_67rice impatient_250p200_67rice fem_married $fullset frac_veryown_inc_0_25 frac_veryown_inc_25_50 frac_veryown_inc_50_75 frac_veryown_inc_75_100 active if group=="T" & call==1 & reached==1 & female==1, robust

log off
*/


gen surveyed = 1 if call==1
replace surveyed = 0 if call ==.
*gen dist_GB = dbutuan if cid ==cid2
replace dist_GB = dampayon if butuan ==0
destring pop, ignore(",") replace
*gen brgy_penetration = no_clients /pop
*bysort brgy: egen sd_totbal = sd(totbal)
*bysort brgy: egen mean_totbal = mean(totbal)

*log using appendix1.log, replace
/*Appendix 1*/
*PanelA
ttest dist_GB, by(surveyed)
ttest totbal, by(surveyed)
ttest active, by(surveyed)
ttest brgy_penetration, by(surveyed)
ttest mean_totbal, by(surveyed)
ttest sd_totbal, by(surveyed)
ttest pop, by(surveyed)

*PanelB
tab group surveyed, row

*PanelC
tabstat balchange if group =="T", by(surveyed) stats(mean sem)
tabstat balchange if group =="M", by(surveyed) stats(mean sem)
tabstat balchange if group =="C", by(surveyed) stats(mean sem)
tabstat balchange, by(surveyed) stats(mean sem)

log close


