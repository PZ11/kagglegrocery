
##############################################################################
############## Set PromoFlag to zero on Continue Promos ######
##############################################################################

 				
sel item_nbr, store_nbr				
from 				
( 				
-- promo week is more than 60% of total 				
sel  item_nbr, store_nbr, 				
	max(salesdate) as max_date, 			
	min(salesdate) as min_date  ,			
	max(salesdate) -  min(salesdate) + 1  as daycount, 			
	count(*) as recordcount ,			
	average(unit_sales) as avg_sales,			
	sum(onpromotion) as promo_wks			
from  pzhang.g_train s				
-- where (s.salesdate) > '2017-01-01' 				
having 				
--and max_date > '2017-07-15'				
promo_wks > recordcount * 0.5				
group by 1,2  				
) a				
				
UNION 				
				
sel item_nbr, store_nbr				
from 				
( 				
-- New SKU after May 51				
sel  item_nbr, store_nbr, 				
	max(salesdate) as max_date, 			
	min(salesdate) as min_date  ,			
	max(salesdate) -  min(salesdate) + 1  as daycount, 			
	count(*) as recordcount ,			
	average(unit_sales) as avg_sales,			
	sum(onpromotion) as promo_wks			
from  pzhang.g_train s				
-- where (s.salesdate) > '2017-01-01' 				
having 				
min_date > '2017-05-01'				
group by 1,2  				
) b				

##############################################################################
############## Set forecast to zero on the missing record for Wed Promo ######
##############################################################################


replace view g_wed_promo
AS 	
sel s.store_nbr, s.item_nbr , 
	count(*) as wed_cnt,  sum(onpromotion) as sum_promo
from  pzhang.g_train  s
inner join pzhang.g_calendar c
on s.salesdate = c.salesdate
where  s.salesdate >= '2017-01-02'
	and daynumber  = 1 
	having wed_cnt = sum_promo
	group by 1 ,2 
;


	sel id , store_nbr,  item_nbr, salesdate    from pzhang.g_test where
	 (store_nbr,  item_nbr)
	in ( select store_nbr,  item_nbr from g_wed_promo) 
	and salesdate in ( '2017-08-16', '2017-08-23', '2017-08-30' ) 
	and onpromotion = 0 
;



##############################################################################
CREATE TABLE pzhang. g_continue_promo_s_i
as ( 
	sel 
	item_nbr, store_nbr, min(salesdate) as startdate, max(salesdate) as enddate 
	from  pzhang.g_test
	where (item_nbr, store_nbr) in (
		sel a.item_nbr, a.store_nbr
		from 
		(
			sel  item_nbr, store_nbr, sum(onpromotion) as sum_promo 	 
			from  pzhang.g_test s
			inner join pzhang.g_calendar c
			on s.salesdate = c.salesdate
			where s.salesdate > '2017-08-15' 
			having sum_promo between 11 and 15 
			group by 1, 2 
		) a , 
		(
			sel  item_nbr, store_nbr, sum(onpromotion) as sum_promo 	 
			from  pzhang.g_test s
			inner join pzhang.g_calendar c
			on s.salesdate = c.salesdate
			where s.salesdate between '2017-08-16'  and '2017-08-18'
			having sum_promo >=  2 
			group by 1, 2 
		) b 
		where a.store_nbr = b.store_nbr
		and a.item_nbr = b.item_nbr 
	)  
	and onpromotion = 1 
	group by 1,2 
) WITH DATA 
primary index (	item_nbr, store_nbr )  
;

SELECT t.id, t. store_nbr, t. item_nbr, salesdate   FROM  pzhang.g_test t
INNER JOIN pzhang. g_continue_promo_s_i p
ON t.store_nbr = p.store_nbr
		and t.item_nbr = p.item_nbr 
		and t.salesdate between p.startdate and p.enddate 
ORDER BY t. store_nbr, t. item_nbr, salesdate 

WHERE t.onpromotion = 0 

##############################################################################
##############################################################################
drop table g_ratio_ty_lm_2017 ; 
drop table g_ratio_ly_tm_2017 ; 

create table g_ratio_ty_lm_2017
as (
select a.store_nbr, a.item_nbr, sales_ln_m5/sales_ln_m4 as ratio_ty_lm
from 
(
	sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m5
	from pzhang.g_expand e, 
	pzhang.g_calendar c
	where c.fiscalyear = 2017
	and  e.salesdate = c.salesdate 
	 and fiscalperiod = 5 
	group by 1,2
) a inner join 
(
	sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m4
	from pzhang.g_expand e, 
	pzhang.g_calendar c
	where c.fiscalyear = 2017
	and  e.salesdate = c.salesdate 
	 and fiscalperiod = 4
	group by 1,2
	having sales_ln_m4 >0 
) b
on a.store_nbr = b.store_nbr
and a.item_nbr = b.item_nbr 
) with data 
primary index( item_nbr ) 
;


create table g_ratio_ly_tm_2017
as (
	select a.store_nbr, a.item_nbr, sales_ln_m6/sales_ln_m5 as ratio_ly_tm
	from 
	(
		sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m6
		from pzhang.g_expand e, 
		pzhang.g_calendar c
		where c.fiscalyear = 2016
		and  e.salesdate = c.salesdate 
		 and fiscalperiod = 6
		group by 1,2
	) a inner join 
	(
		sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m5
		from pzhang.g_expand e, 
		pzhang.g_calendar c
		where c.fiscalyear = 2016
		and  e.salesdate = c.salesdate 
		 and fiscalperiod = 5
		group by 1,2
		having sales_ln_m5 >0 
	) b
	on a.store_nbr = b.store_nbr
	and a.item_nbr = b.item_nbr 
) with data 
primary index( item_nbr ) 
;


