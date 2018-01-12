

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


