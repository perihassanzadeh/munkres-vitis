//Based on: https://github.com/SyllogismRXS/openmht/tree/master
#include <iostream>
#include <list>
#include <map>
#include <bits/stdc++.h>

#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <typeinfo>

#include <Eigen/Dense>

#include <openmht/multi/MHT.h>
#include <openmht/plot/Plot.h>

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/random/normal_distribution.hpp>

using std::cout;
using std::endl;
using std::string;
using std::vector;

class Contact {
public:
     int id;
     Eigen::Vector2d position;
     Eigen::Vector2d velocity;
};

// Random Number Generation -- rng_normal()
boost::mt19937 gener_; 
boost::normal_distribution<> normal_dist_(0,1.0);
boost::variate_generator<boost::mt19937&, 
                         boost::normal_distribution<> > rng_normal(gener_,normal_dist_);

void step_dynamics(std::list<Contact> &contacts, double dt)
{
     for (std::list<Contact>::iterator it = contacts.begin(); 
          it != contacts.end(); it++) {
          it->position = it->position + it->velocity*dt;
     }
}

Contact add_noise(Contact &contact)
{
     Contact c = contact;          
     c.position(0,0) = c.position(0,0) + rng_normal();
     c.position(1,0) = c.position(1,0) + rng_normal();          
     return c;
}

/*
readCSV - Parses csv file containing rows formatted with timestamp, x pos, y pos
Input - Target timestamp
Returns vector of coordinate pairs for specified timestamp
*/
/*
vector<vector<double>> readCSV(int target_timestamp)
{
     std::ifstream infile("/home/peri/Desktop/detections.csv");

     vector<vector<double>> coords;
     string line;

     while(getline(infile, line))
     {
          std::stringstream strparse(line);
          string temp;
          vector<double> temp_vec;

          double xpos, ypos;
          int timestamp;

          int count = 1;
          while(getline(strparse, temp, ','))
          {
               temp.erase(remove(temp.begin(), temp.end(), ' '), temp.end());

               if(count==1){
                    timestamp = stoi(temp);
               }
               else if(count==2){
                    xpos=stod(temp);
               }
               else if(count ==3){
                    ypos = stod(temp);
               }
               else{
                    cout<< "Invalid count" << endl;
               }
               
               count++;
          }

          temp_vec.push_back(xpos);
          temp_vec.push_back(ypos);
          //cout << timestamp << endl;
          //cout << xpos << endl;
          //cout << yposfalse << endl;

          if(timestamp==target_timestamp)
          {
               coords.push_back(temp_vec);
               //cout << temp_vec[0] << endl;
          }          
     }

     return coords;
}
*/

int main(int argc, char *argv[])
{
     //Read in detections from CSV
     //vector<vector<double>> timestamp_detections;
     //timestamp_detections = readCSV(2);

     //For timing purposes
     clock_t start, end;
     bool plotter = false;

     //Initialize random number generator
     rng_normal.engine().seed(static_cast<unsigned int>(std::time(0)));
     rng_normal.distribution().reset();
     
     start  = clock();

     //Initial Points
     int num_contacts = 5;   
     std::list<Contact> contacts;
     for (int i = 0; i < num_contacts; i++) {
          Contact c;     
          c.id = i;
          c.position << rng_normal()*10, rng_normal()*10;
          c.velocity << rng_normal()*1, rng_normal()*1;
          contacts.push_back(c);
     }

     /*
     for (int i = 0; i < 100; i++) {
          cout << rng_normal() << endl;
     }
     */
     double t0 = 0;
     double dt = 1;
     double tend = 10;
     
     openmht::Plot plot;
     std::map<int, std::list<Eigen::Vector2d> > truth, measured, tracked;
          
     openmht::MHT mht;   
     mht.set_dt(dt);
     for (double t = t0; t < tend; t += dt) {

          // Convert from your list of 2D detection points to a list of
          std::list<openmht::Measurement> m_list;
          for (std::list<Contact>::iterator it = contacts.begin(); 
               it != contacts.end(); it++) {

               Contact c = add_noise(*it);
               
               openmht::Measurement m;
               m.set_position(c.position);
               m_list.push_back(m);

               // Save the measured position for plotting later
               measured[c.id].push_back(c.position);
          }
          
          // Process the list of measurements with MHT
          mht.process_measurements(m_list);

          // Get a list of the fused entities / contacts:
          std::list<openmht::Entity> ents = mht.entities();

          // Save the filtered locations of each track
          for (std::list<openmht::Entity>::iterator it = ents.begin(); 
               it != ents.end(); it++) {            
               tracked[it->id()].push_back(it->position());
          }                              
          
          // Save the truth tracks:
          for (std::list<Contact>::iterator it = contacts.begin(); 
               it != contacts.end(); it++) {
               truth[it->id].push_back(it->position);
          }

          // Plot the results:
          
          if(plotter==true)
          {
               std::string title = "Tracks";
               std::list< std::map<int, std::list<Eigen::Vector2d> > > lists;
               std::list<std::string> labels;
               std::list<std::string> styles;
               std::string options = "";
               std::list<std::string> objects;
               
               lists.push_back(truth);
               labels.push_back("Truth");
               styles.push_back("linespoints");
               
               lists.push_back(measured);
               labels.push_back("Measured");
               styles.push_back("linespoints");
               
               lists.push_back(tracked);          
               labels.push_back("Tracked");
               styles.push_back("linespoints");

               plot.plot(lists, title, labels, styles, options, objects, false);
          }          
          
          // Step dynamics:
          step_dynamics(contacts,dt);
     }
     cout << "Truth lenght: " << truth.size() << endl;
     cout << "Measured length: " << measured.size() << endl;
     cout << "Tracked length: " << tracked.size() << endl;

     end = clock();
     double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
     cout << "Execution Time:" << time_taken << endl;
     
     if(plotter==true){
          plot.wait();
     }
}

