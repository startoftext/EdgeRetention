# EdgeRetention

This is an attempt to predict edge retention for knives based on steal composition. I am using cut test data from a youtuber named
Cedric and Ada who cuts sisal rope with various knife steal types until the blade will no longer slice paper cleanly. You can find his youtube channel here https://www.youtube.com/channel/UCdICfnpxD9uzHLaSr3DN55g and he shared his data in a goodle sheet here https://docs.google.com/spreadsheets/d/1b_rNfdJnL9oyn-JoL9yUHhUmDLAP1hJ1dN_0q5G4tug/edit#gid=642985315 . 

I augmented his data with the steel composition percentages and removed some test cases which I could not find data for,
or for which data did not make sense such as tungsten carbide or ceramic. Some limmitations of the data are that I do
not have hardness or stock thickness for all theses cases. Also many of these tests are missing the edge angle that they are sharpened to.
