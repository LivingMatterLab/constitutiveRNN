# constitutiveRNN
Code and data for automated viscoelastic model discovery for muscle tissue.

Scientific paper using this code: 

<b> Files: </b> <br>
1. Automated_Analysis: Performs automated template matching to identify microtubules in the full image set. <br>
2. Manual_Edits: Allows the user to make edits to the automated results by clicking missed microtubules and drawing boxes around false positives. <br>
3. Registration: Registers microtubule match locations in adjacent images to trace microtubules throughout the full image stack. <br>
4. RealisticGeometry_ABAQUSInput: Adds randomly dispersed crosslinks to the ssTEM-based microtubule geometry and converts the results into an ABAQUS mesh. Generates an ABAQUS input file. <br> 
5. IdealizedGeometry_ABAQUSInput: Generates an idealized geometry of regularly spaced, equally-sized microtubules with randomly dispersed crosslinks. Generates an ABAQUS input file. <br>
