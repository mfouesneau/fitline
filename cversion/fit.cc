#include <iostream>
#include <iomanip>
#include <vector>
#include <istream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <stdlib.h>
#include "mfstats.hh"
#include "mfopts.hh"


/**
 *  Data Class:  Container to a dataset obtain from an ascii file
 */
template<class T> 
class Data: public std::vector<std::vector<T> > {

public:

    /* Attributes */
    std::vector<size_t> shape = {0,0};  /** shape of the dataset (nlines, ncols) */
    size_t nlines = 0;                  /** number of lines in the dataset */
    size_t ncols  = 0;                  /** number of columns in the dataset */

    /* methods */

    /**
     * Data::load_matrix  Load an ascii file into a matrix (vectors of vectors)
     *
     * @param is        stream to read-in
     * @param matrix    matrix to fill
     * @param delim     column delimiter(default '\t')
     * @param comment   comment line marker
     */
    void load_matrix(std::istream& is, std::vector< std::vector<T> >& matrix,
        std::string delim=" ", std::string comment="#"){

        std::string line;
        std::string strnum;

        // clear first
        matrix.clear();
        strnum.clear();
        line.clear();

        while(getline(is, line)){
            // skip empty lines
            if(line.size() == 0) {continue;}
            // skip comment lines
            if (line.substr(0, comment.size()) == comment){continue;}
            // add an empty line into the matrix
            matrix.push_back(std::vector<T>());
            for (std::string::const_iterator i = line.begin(); i != line.end(); ++ i){
                // If i is not a delimiter, then append it to strnum
                if (delim.find(*i) == std::string::npos)
                {
                strnum += *i;
                // if this is the last char of the line, continue
                if (i + 1 != line.end()) {continue;}
                }

                // if strnum is still empty, it means the previous char is also a
                // delim (several delims appear together). Ignore this char.
                if (strnum.empty()) {continue;}

                // If we reach here, we got a number. Convert it to double.
                T number;
                std::istringstream(strnum) >> number;
                matrix.back().push_back(number);
                strnum.clear();
            }
        }
    };


    /**
     * Data::Data Constructor from a filename
     *
     * @param fname     filename to open and read-in
     */
    Data(std::string fname){

        // read the file
        std::ifstream is(fname);

        std::vector< std::vector<T> > mat;

        // load the matrix
        load_matrix(is, mat);
        this->nlines = mat.size();
        this->clear();
        for(size_t i=0; i<mat.size(); ++i){
                // std::vector<T> col = mat[i];
                // std::vector<T> newcol(col.size());
                // for (size_t j=0; j<col.size(); ++j){
                //         newcol[j] = col[j];
                // }
                this->push_back(mat[i]);
        }
        this->ncols = mat[0].size();
        this->shape[0] = this->nlines;
        this->shape[1] = this->ncols;

    };


    /**
     * Data::clean_data:  Clean data for unreliable uncertainties and replace
     *                    them by a fraction of the value.
     *
     * @param null_threshold    limit under which uncertainty is too small to believe
     * @param xfloor            percentage fraction of x
     * @param yfloor            percentage fraction of y
     */
    void clean_data(T null_threshold=1e-6,
                    T xfloor=10.,
                    T yfloor=10.){

        for (size_t i=0; i<this->nlines; ++i){
            std::vector<T> d = (*this)[i];
            T x    = d[0];
            T y    = d[1];
            T xmin = d[2];
            T xmax = d[3];
            T ymin = d[4];
            T ymax = d[5];
            if (ymin < null_threshold)
                    ymin += yfloor * 0.01 * y;
            if (ymax < null_threshold)
                    ymax += yfloor * 0.01 * y;
            if (xmax < null_threshold)
                    xmax += xfloor * 0.01 * x;
            if (xmin < null_threshold)
                    xmin += xfloor * 0.01 * x;

            // update values
            d[0] = x;
            d[1] = y;
            d[2] = xmin;
            d[3] = xmax;
            d[4] = ymin;
            d[5] = ymax;
        }
    };

    /**
     * Print the dataset on stdout
     *
     * @param Nchar number of characters per column (default:12)
     */
    void print(int Nchar = 12){

        // print out the matrix
        for (size_t i = 0; i < this->nlines; ++i) {
            for (size_t j = 0; j < this->ncols; ++j) {
                std::cout << std::setw(Nchar) << std::left << (*this)[i][j] << " ";
            }
            std::cout << std::endl;
        }
    };


    /**
     * Print the first few lines from the dataset on stdout
     *
     * @param N     number of lines to show (default:5)
     * @param Nchar number of characters per column (default:12)
     */
    void print_head(size_t N=5, int Nchar = 12){
        for (size_t i = 0; (i < this->nlines) & (i < N); ++i) {
            for (size_t j = 0; j < this->ncols; ++j) {
                std::cout << std::setw(Nchar) << std::left << (*this)[i][j] << " ";
            }
            std::cout << std::endl;
        }
        for (size_t j = 0; j < this->ncols; ++j) {
            std::cout << std::setw(Nchar) << std::left << "..." << " ";
        }
        std::cout << std::endl;
    };
};

/*-----------------------------------------------------------------------------
 *  Data and model manipulation
 *-----------------------------------------------------------------------------*/


/**
 * sample data from the values and uncertainties
 *
 *  @param d            Data instance
 *  @param data         [x,y] arrays of values to return
 *  @param generator    random generator instance
 *  @param errorfloor   minimum uncertainty to assure in data units (default:1e-4)
 *  @param bootstrap    set to bootstrap the data (default:false)
 */
void sample_data_pdf(Data<double>& d, std::vector<std::vector<double> >& data,
        std::mt19937& generator,
        double errorfloor=1e-4,
        bool bootstrap=false){

    // select points either all or randomly
    std::vector<int> index(d.size());
    if (bootstrap){
        std::uniform_real_distribution<float> randint(0, d.size() - 1);
        for (size_t i=0; i<d.size(); ++i){
            index[i] = randint(generator);
        }
    } else {
        for (size_t i=0; i<d.size(); ++i){
            index[i] = i;
        }
    }

    // sample
    std::vector<double> xdata(d.size());
    std::vector<double> ydata(d.size());

    // only used to decide which side to go
    std::uniform_real_distribution<float> side(0,1);

    for (size_t i = 0; i < d.size(); ++i) {
        // select random side in both direction

        size_t ind = index[i];
        double x = d[ind][0];
        double y = d[ind][1];
        double xmin = d[ind][2];
        double xmax = d[ind][3];
        double ymin = d[ind][4];
        double ymax = d[ind][5];

        if (side(generator) > 0.5){
            std::normal_distribution<double> yrand(0., ymax + errorfloor);
            ydata[i] = y + std::fabs(yrand(generator));
        } else {
            std::normal_distribution<double> yrand(0., ymin + errorfloor);
            ydata[i] = y - std::fabs(yrand(generator));
        }
        if (side(generator) > 0.5){
            std::normal_distribution<double> xrand(0., xmax + errorfloor);
            xdata[i] = x + std::fabs(xrand(generator));
        } else {
            std::normal_distribution<double> xrand(0., xmin + errorfloor);
            xdata[i] = x - std::fabs(xrand(generator));
        }
    }

    // store values for return
    data.clear();
    data.push_back(xdata);
    data.push_back(ydata);
}


/**
 *  pass data into log10 values
 *
 *  @param data         the [x,y] dataset
 *  @param log_xnorm    normalization of x values
 *  @param log_ynorm    normalization of y values
 */
void transform_data_sample_to_log (std::vector<std::vector<double> >& data,
                double log_xnorm=0.,
                double log_ynorm=0.) {

    size_t N = data[0].size();

    std::vector<double> x = data[0];
    std::vector<double> y = data[1];

    // transform values
    for (size_t  i = 0; i < N; ++i) {
        data[0][i] = std::log10(x[i]) - log_xnorm;
        data[1][i] = std::log10(y[i]) - log_ynorm;
    }
}


/**
 *  computes the Orthogonal Least-Square maximum a posteriori
 *
 *  @param data     [x, y] value arrays
 *  @param theta    [slope, intercept, dispersion] values to fill
 */
void OSL_MAP(std::vector<std::vector<double> >& data,
             std::vector<double>& theta){

    // calculate mean values
    // std::cout << data[0][0] << "  " << data[1][0];
    double mx = nan_mean<double>(data[0]);
    double my = nan_mean<double>(data[1]);
    // std::cout << "mx, my " << mx << "  " << my << std::endl;

    // calculate n * variances and n-1 * covariance
    size_t Nvalid = 0;
    for(size_t i=0; i<data[0].size(); ++i){
            if (!std::isnan(data[0][i]) && !std::isnan(data[1][i])){
                    Nvalid++;
            }
    }
    // std::cout << count_notnan(xdata, ydata) << std::endl;
    double sx = nan_var<double>(data[0]) * (double) Nvalid;
    double sy = nan_var<double>(data[1]) * (double) Nvalid;
    double sxy = nan_cov<double>(data[0], data[1]) * (double)(Nvalid - 1);

    // get slope a and offset b:  y| a, b ~ a * x + b
    double a = 0.5 * (sy - sx +
                      std::sqrt(
                          (sy - sx) * (sy - sx) + 4. * sxy * sxy)
                          ) / sxy;
    double b = my - a * mx;

    // compute the Orthogonal distance to that line
    // (vx, vy) is the orthogonal vector.
    double vx = -a / std::sqrt( 1. + a * a);
    double vy = -1. / std::sqrt( 1. + a * a);
    // for each (x, y) we need to do the dot product of (x - xm, y - ym) . (vx, vy)
    double rms = 0;
    size_t N = data.size();
    for (size_t  i = 0; i < N; ++i) {
        double xi = data[0][i];
        double yi = data[1][i];
        if (!std::isnan(xi) && !std::isnan(yi)){
            rms += std::pow(((xi - mx) * vx) + ((yi - my) * vy), 2);
        }
    }
    rms = rms / (double) (Nvalid - 1); // rms to variance

    theta.clear();
    theta.push_back(a);
    theta.push_back(b);
    theta.push_back(rms);
}



/**
 *  Generate mock data to test the model
 *
 *  @param N    size of the data
 *  @param a    slope of the model
 *  @param b    intercept of the model
 *  @param d    dispersion perpendicular to the model
 *  @param data dataset newly created
 *  @param gen  random generator
 */
void mock_affine_data(size_t N, double a, double b, double d,
                      std::vector<std::vector<double> >& data,
                      std::mt19937& gen){
    data.clear();
    std::uniform_real_distribution<double> xrand(0,1);
    // d^2 = sigma ^ 2 + sigma ^ 2 = 2 * sigma ^ 2
    // ==> sigma = sqrt(d^2 / 2)
    double sigma = std::sqrt(d * d * 0.5);
    std::normal_distribution<double> drand(0, sigma);

    std::vector<double> x(N);
    std::vector<double> y(N);

    for(size_t i=0; i<N; ++i){
        x[i] = xrand(gen) - 0.5 + drand(gen);
        y[i] = a * x[i] + b + drand(gen);
    }

    data.push_back(x);
    data.push_back(y);
}


/*=============================================================================
 *  Main functions: mock data or real data
 *=============================================================================*/


/**
 *  main_mock:  Generate mock data and perform the fitting
 */
int main_mock(
    // sampling from P(data)
    size_t number_of_samples = 10000,
    //mock tests
    size_t Ndata           = 100, // only for mockdata
    double mock_slope      = 4.,
    double mock_intercept  = 0.,
    double mock_dispersion = 1e-2)
{
    using namespace std;

    cout << "Mocking data" << endl;
    cout << "  + N data points: " << Ndata << endl;
    cout << "  + Model (a, b, d) = " << mock_slope
         << "," << mock_intercept
         << "," << mock_dispersion << endl;

    cout << "Performing fit" << endl;
    cout << "  + number of samples from p(D) " << number_of_samples << endl;

    //generate sub-dataset
    std::vector<std::vector<double> > data;
    std::vector<double> theta;

    //get a random generator
    // std::default_random_engine generator;
    std::random_device rd;
    std::mt19937 generator(rd());


    // store parameters
    std::vector<double> a(number_of_samples);
    std::vector<double> b(number_of_samples);
    std::vector<double> d(number_of_samples);

    for (size_t i = 0; i < number_of_samples; ++i) {

        mock_affine_data(Ndata, mock_slope, mock_intercept,
                        mock_dispersion, data, generator);

        OSL_MAP(data, theta);

        a[i] = theta[0];
        b[i] = theta[1];
        d[i] = theta[2];
    }

    cout << "Results" << endl;
    // percentiles
    std::vector<double> perc = {15.87, 50, 84.13};

    std::vector<double> pa = percentile(a, perc);
    std::vector<double> pb = percentile(b, perc);
    std::vector<double> pd = percentile(d, perc);

    cout << "  + a = " << std::setw(12) << pa[1] 
        << " | + " << std::setw(12) << pa[2] - pa[1] 
        << " - "   << std::setw(12) << pa[1] - pa[0] << endl;
    cout << "  + b = " << std::setw(12) << pb[1] 
        << " | + " << std::setw(12) << pb[2] - pb[1] 
        << " - "   << std::setw(12) << pb[1] - pb[0] << endl;
    cout << "  + d = " << std::setw(12) << pd[1] 
        << " | + " << std::setw(12) << pd[2] - pd[1] 
        << " - "   << std::setw(12) << pd[1] - pd[0] << endl;

    return 0;
}

/**
 * main_data:  Read ascii file and perform the fitting
 */
int main_data(
    // sampling from P(data)
    std::string fname = "reff_NSC_Mass_late.dat",
    size_t number_of_samples = 10000,
    double errorfloor = 1e-4,
    bool bootstrap = true,
    // normalazing dimensions
    double log_xnorm = 6.,
    double log_ynorm = 1.,
    // cleaning
    double xfloor = 10,
    double yfloor = 10,
    double null_threshold=1e-6)
{
    /*
     * parameters
     */

    using namespace std;

    // load data and clean it for bad uncertainties
    cout << "loading data file: " << fname << endl;
    Data<double> mat(fname);
    cout << "  + cleaning data" << endl;
    mat.clean_data(null_threshold, xfloor, yfloor);
    cout << "  + final shape = " << mat.shape[0] << "," << mat.shape[1] << endl;
    cout << "Performing fit" << endl;
    cout << "  + number of samples from p(D) " << number_of_samples << endl;
    cout << "  + bootstrap data " << bootstrap << endl;
    cout << "  + log_norm(x, y) = " << log_xnorm << "," << log_ynorm << endl;
    cout << "  + errorfloor = " << errorfloor << endl;

    //generate sub-dataset
    std::vector<std::vector<double> > data;
    std::vector<double> theta;

    //get a random generator
    // std::default_random_engine generator;
    std::random_device rd;
    std::mt19937 generator(rd());


    // store parameters
    std::vector<double> a(number_of_samples);
    std::vector<double> b(number_of_samples);
    std::vector<double> d(number_of_samples);

    for (size_t i = 0; i < number_of_samples; ++i) {
        sample_data_pdf(mat, data, generator, errorfloor, bootstrap);
        transform_data_sample_to_log(data, log_xnorm, log_ynorm);
        OSL_MAP(data, theta);

        a[i] = theta[0];
        b[i] = theta[1];
        d[i] = theta[2];
    }

    cout << "Results" << endl;
    // percentiles
    std::vector<double> perc = {15.87, 50, 84.13};

    std::vector<double> pa = percentile(a, perc);
    std::vector<double> pb = percentile(b, perc);
    std::vector<double> pd = percentile(d, perc);

    cout << "  + a = " << std::setw(12) << pa[1] 
        << " | + " << std::setw(12) << pa[2] - pa[1] 
        << " - "   << std::setw(12) << pa[1] - pa[0] << endl;
    cout << "  + b = " << std::setw(12) << pb[1] 
        << " | + " << std::setw(12) << pb[2] - pb[1] 
        << " - "   << std::setw(12) << pb[1] - pb[0] << endl;
    cout << "  + d = " << std::setw(12) << pd[1] 
        << " | + " << std::setw(12) << pd[2] - pd[1] 
        << " - "   << std::setw(12) << pd[1] - pd[0] << endl;

    return 0;
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  main function
 * =====================================================================================
 */
int main(int argc, char *argv[])
{
    using namespace std;

    // default options
    // ===============
    bool run_mock = false;
    // mock data options
    size_t Ndata           = 100; // only for mockdata
    double mock_slope      = 4.;
    double mock_intercept  = 0.;
    double mock_dispersion = 1e-2;
    
    // sampling from P(data)
    size_t number_of_samples = 10000;
    double errorfloor = 1e-4;
    bool bootstrap = true;
    // normalazing dimensions
    double log_xnorm = 6.;
    double log_ynorm = 1.;
    // cleaning
    double xfloor = 10;
    double yfloor = 10;
    double null_threshold=1e-6;
    std::string fname = "";

    // parse options
    // =============
    
    mfopts::Options opt(argv[0], "");

    opt.add_option("-h,--help", "Display help message");

    // mock data options
    opt.add_option("--mock"   , "Set to run mock data sampling");
    opt.add_option("--mock_N" , "Number of data points in mock sample");
    opt.add_option("--mock_a" , "slope of the mock data");
    opt.add_option("--mock_b" , "intercept of the mock data");
    opt.add_option("--mock_d" , "intrinsic dispersion of the mock data");

    // real data file options

    opt.add_option("-i,--input" , "Input data file");
    opt.add_option("-n,--nsamples" , 
            "Number of samples from data to generate the probability distribution");
    opt.add_option("-b,--bootstrap", "set to bootstrap the data (not only their likelihood)");
    opt.add_option("--errorfloor" , "threshold under which errors are not reliable");
    opt.add_option("--logxnorm" , "x-data normalization value");
    opt.add_option("--logynorm" , "y-data normalization value");
    opt.add_option("--xfloor" , "floor of x-value uncertainty (in %)");
    opt.add_option("--yfloor" , "floor of y-value uncertainty (in %)");

    opt.parse_options(argc, argv);

    if (opt.has_option("--help"))
    {
        std::cout << opt.help() << std::endl;
        exit(0);
    }

    if (opt.has_option("--mock"))
        run_mock = true;
    if (opt.has_option("--mock_N"))
        Ndata = opt.get_option<size_t>("--mock_N");
    if (opt.has_option("--mock_a"))
        mock_slope = opt.get_option<double>("--mock_a");
    if (opt.has_option("--mock_b"))
        mock_intercept = opt.get_option<double>("--mock_b");
    if (opt.has_option("--mock_d"))
        mock_dispersion = opt.get_option<double>("--mock_d");
    if (opt.has_option("--nsamples"))
        number_of_samples = opt.get_option<size_t>("--nsamples");
    if (opt.has_option("--errorfloor"))
        errorfloor = opt.get_option<double>("--errorfloor");
    if (opt.has_option("--logxnorm"))
        log_xnorm = opt.get_option<double>("--logxnorm");
    if (opt.has_option("--logynorm"))
        log_ynorm = opt.get_option<double>("--logynorm");
    if (opt.has_option("--xfloor"))
        xfloor = opt.get_option<double>("--xfloor");
    if (opt.has_option("--yfloor"))
        yfloor = opt.get_option<double>("--yfloor");
    if (opt.has_option("--input"))
        fname = opt.get_option<std::string>("--input");

    // run
    // ====
    if (run_mock){
        main_mock(number_of_samples, Ndata, mock_slope, mock_intercept, mock_dispersion);
    } else {
        if (fname == ""){
            std::cout << "You must specify a file to work with (-i option)" << std::endl;
            exit(1);
        }
        main_data(fname, number_of_samples, errorfloor, bootstrap, log_xnorm,
                log_ynorm, xfloor, yfloor, null_threshold);
    }
    return 0;
}

// vim: expandtab:ts=4:softtabstop=4:shiftwidth=4
