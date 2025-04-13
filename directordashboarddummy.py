import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

def display_director_dashboard():
    st.header("University Director Dashboard")
    
    # Simple authentication
    with st.expander("Director Authentication", expanded=True):
        director_password = st.text_input("Enter Director Access Code", type="password")
        authenticate = st.button("Login")
        
        if authenticate:
            if director_password == "admin123":  # In production, use a more secure method
                st.session_state['director_authenticated'] = True
                st.success("Authentication successful!")
            else:
                st.error("Invalid access code")
                st.session_state['director_authenticated'] = False
    
    # Only show dashboard if authenticated
    if st.session_state.get('director_authenticated', False):
        display_dashboard_content()
    else:
        st.info("Please authenticate to view the director dashboard")

def display_dashboard_content():
    """Display dashboard content with realistic dummy data"""
    
    # Generate dummy metrics
    metrics = {
        "total_applications": 1247,
        "processed_applications": 983,
        "processing_percentage": 78.8,
        "eligible_candidates": 645,
        "shortlisted_candidates": 420,
        "status_distribution": {
            "Submitted": 264,
            "Processing": 187,
            "Eligible": 645,
            "Shortlisted": 420,
            "Rejected": 351
        },
        "category_distribution": {
            "General": 587,
            "OBC": 312,
            "SC": 198,
            "ST": 98,
            "EWS": 52
        },
        "rank_distribution": {
            "ranges": ["1-1000", "1001-5000", "5001-10000", "10001+"],
            "counts": [85, 420, 512, 230]
        },
        "loan_applications": 387,
        "approved_loans": 245,
        "total_loan_amount": 68250000,
        "loan_status_distribution": {
            "Approved": 245,
            "Pending": 89,
            "Rejected": 53
        }
    }
    
    # Generate dummy recent applications with dates
    def random_date(start_date, end_date):
        return start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds())))
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)
    
    recent_applications = pd.DataFrame({
        'Application ID': [f'S2023-{i:04d}' for i in range(456, 446, -1)],
        'Name': ['Rahul Sharma', 'Priya Patel', 'Amit Singh', 'Neha Gupta', 'Vikram Joshi', 
                'Ananya Reddy', 'Karthik Nair', 'Divya Menon', 'Arjun Kapoor', 'Meera Desai'],
        'Exam Type': ['JEE Main', 'WBJEE', 'JEE Main', 'WBJEE', 'JEE Main', 
                     'WBJEE', 'JEE Main', 'WBJEE', 'JEE Main', 'WBJEE'],
        'Exam Rank': [1245, 387, 5421, 892, 2107, 156, 3789, 654, 4321, 987],
        'Category': ['General', 'OBC', 'SC', 'General', 'EWS', 
                    'General', 'OBC', 'ST', 'SC', 'General'],
        'Status': ['Shortlisted', 'Eligible', 'Processing', 'Shortlisted', 'Rejected',
                  'Shortlisted', 'Eligible', 'Processing', 'Rejected', 'Shortlisted'],
        'Submission Date': [random_date(start_date, end_date) for _ in range(10)],
        'Document Status': ['Verified', 'Verified', 'Pending', 'Verified', 'Verified',
                          'Verified', 'Verified', 'Pending', 'Verified', 'Verified'],
        'Loan Status': ['Approved', 'Pending', 'N/A', 'Approved', 'Rejected',
                       'Approved', 'Pending', 'N/A', 'Rejected', 'Approved']
    })
    
    # Format dates
    recent_applications['Submission Date'] = recent_applications['Submission Date'].dt.strftime('%Y-%m-%d')
    
    # Overview section
    st.subheader("Admission Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", f"{metrics['total_applications']:,}")
    with col2:
        st.metric("Applications Processed", 
                 f"{metrics['processed_applications']:,}",
                 f"{metrics['processing_percentage']}%")
    with col3:
        st.metric("Eligible Candidates", f"{metrics['eligible_candidates']:,}")
    with col4:
        st.metric("Shortlisted", f"{metrics['shortlisted_candidates']:,}")
    
    # Application status breakdown
    st.subheader("Application Status Breakdown")
    
    # Create pie chart for status distribution
    status_df = pd.DataFrame({
        "Status": list(metrics["status_distribution"].keys()),
        "Count": list(metrics["status_distribution"].values())
    })
    
    fig_status = px.pie(status_df, 
                       values="Count", 
                       names="Status",
                       title="Application Status Distribution",
                       color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Applications by department/category
    st.subheader("Applications Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Category distribution bar chart
        category_df = pd.DataFrame({
            "Category": list(metrics["category_distribution"].keys()),
            "Count": list(metrics["category_distribution"].values())
        })
        
        fig_category = px.bar(category_df, 
                             x="Category", 
                             y="Count",
                             title="Applications by Category",
                             color="Category",
                             text="Count")
        fig_category.update_traces(textposition='outside')
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Rank distribution bar chart
        rank_df = pd.DataFrame({
            "Rank Range": metrics["rank_distribution"]["ranges"],
            "Count": metrics["rank_distribution"]["counts"]
        })
        
        fig_rank = px.bar(rank_df, 
                         x="Rank Range", 
                         y="Count",
                         title="Exam Rank Distribution",
                         color="Rank Range",
                         text="Count")
        fig_rank.update_traces(textposition='outside')
        st.plotly_chart(fig_rank, use_container_width=True)
    
    # Loan processing summary
    st.subheader("Loan Processing Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Loan Applications", f"{metrics['loan_applications']:,}")
        st.metric("Approved Loans", f"{metrics['approved_loans']:,}")
        st.metric("Total Loan Amount (₹)", f"₹{metrics['total_loan_amount']:,}")
    
    with col2:
        # Loan status pie chart
        loan_df = pd.DataFrame({
            "Status": list(metrics["loan_status_distribution"].keys()),
            "Count": list(metrics["loan_status_distribution"].values())
        })
        
        fig_loan = px.pie(loan_df, 
                         values="Count", 
                         names="Status",
                         title="Loan Status Distribution",
                         hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Agsunset)
        st.plotly_chart(fig_loan, use_container_width=True)
    
    # Recent applications table
    st.subheader("Recent Applications")
    
    # Style the dataframe
    def color_status(val):
        if val == 'Shortlisted':
            return 'background-color: #4CAF50; color: white'
        elif val == 'Rejected':
            return 'background-color: #F44336; color: white'
        elif val == 'Eligible':
            return 'background-color: #2196F3; color: white'
        elif val == 'Processing':
            return 'background-color: #FFC107; color: black'
        return ''
    
    styled_df = recent_applications.style.applymap(color_status, subset=['Status'])
    
    st.dataframe(styled_df, 
                column_order=['Application ID', 'Name', 'Exam Type', 'Exam Rank', 
                             'Category', 'Status', 'Submission Date', 'Document Status', 'Loan Status'],
                height=400,
                use_container_width=True)
    
    # Action items section
    st.subheader("Action Items")
    
    action_items = [
        {
            "Title": "Finalize Shortlisted Candidates",
            "Description": "Review and approve the final list of shortlisted candidates",
            "Due Date": "2023-07-15",
            "Priority": "High"
        },
        {
            "Title": "Loan Committee Meeting",
            "Description": "Review pending loan applications and make decisions",
            "Due Date": "2023-07-10",
            "Priority": "Medium"
        },
        {
            "Title": "Prepare Admission Letters",
            "Description": "Generate admission letters for shortlisted candidates",
            "Due Date": "2023-07-20",
            "Priority": "High"
        }
    ]
    
    for item in action_items:
        with st.expander(f"⚠️ {item['Title']} - Due: {item['DueDate']}"):
            st.write(f"**Description:** {item['Description']}")
            st.write(f"**Priority:** {item['Priority']}")
            if st.button(f"Mark as Complete", key=f"complete_{item['Title']}"):
                st.success(f"Action item '{item['Title']}' marked as complete!")

# Run the dashboard
display_director_dashboard()